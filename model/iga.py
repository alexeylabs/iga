import numpy as np
import torch
import torch.nn as nn

from options import IGAConfiguration
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.msg_decoder import MsgDecoder
from model.msg_encoder import MsgEncoder
from vgg_loss import VGGLoss
from noise_layers.noiser import Noiser
import utils


class IGA:
    def __init__(self, configuration: IGAConfiguration, device: torch.device, noiser: Noiser, tb_logger):
        """
        :param configuration: Configuration for the net, such as the size of the input image, number of channels in the intermediate layers, etc.
        :param device: torch.device object, CPU or GPU
        :param noiser: Object representing stacked noise layers.
        :param tb_logger: Optional TensorboardX logger object, if specified -- enables Tensorboard logging
        """
        super(IGA, self).__init__()

        self.encoder_decoder = EncoderDecoder(configuration, noiser).to(device)
        self.discriminator = Discriminator(configuration).to(device)
        self.optimizer_enc_dec = torch.optim.Adam(self.encoder_decoder.parameters())
        self.optimizer_discrim = torch.optim.Adam(self.discriminator.parameters())

        self.msg_encoder = MsgEncoder(configuration.message_length, configuration.message_middle_length).to(device)
        self.msg_decoder = MsgDecoder(configuration.message_middle_length, configuration.message_length).to(device)
        self.optimizer_msg_encoder = torch.optim.Adam(self.msg_encoder.parameters(), lr=1e-4)
        self.optimizer_msg_decoder = torch.optim.Adam(self.msg_decoder.parameters(), lr=1e-4)
        self.msg_coding_loss_weight = 0.01

        if configuration.use_vgg:
            self.vgg_loss = VGGLoss(3, 1, False)
            self.vgg_loss.to(device)
        else:
            self.vgg_loss = None

        self.config = configuration
        self.device = device

        self.bce_with_logits_loss = nn.BCEWithLogitsLoss().to(device)
        self.mse_loss = nn.MSELoss().to(device)

        # Defined the labels used for training the discriminator/adversarial loss
        self.cover_label = 1
        self.encoded_label = 0

        self.tb_logger = tb_logger
        if tb_logger is not None:
            from tensorboard_logger import TensorBoardLogger
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            encoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/encoder_out'))
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            decoder_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/decoder_out'))
            discrim_final = self.discriminator._modules['linear']
            discrim_final.weight.register_hook(tb_logger.grad_hook_by_name('grads/discrim_out'))

    def train_on_batch(self, batch: list):
        """
        Trains the network on a single batch consisting of images and messages
        :param batch: batch of training data, in the form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        images, messages = batch
        images.requires_grad = True

        batch_size = images.shape[0]
        self.encoder_decoder.train()
        self.discriminator.train()

        with torch.enable_grad():

            # inverse gradient attention mask generation for the input image
            if self.config.use_mc:
                messages_enc = self.msg_encoder(messages)
                _, _, decoded_messages = self.encoder_decoder(images, messages_enc)
                decoded_messages = self.msg_decoder(decoded_messages)
            else:
                _, _, decoded_messages = self.encoder_decoder(images, messages)

            mse_loss = self.mse_loss(decoded_messages, messages)
            mse_loss.backward()

            grads = utils.normalize_sigmoid(images.grad)
            images_iga_mask = 1 - grads
            images.grad.zero_()
            images.requires_grad = False

            images.mul_(images_iga_mask)

            # ---------------- Train the discriminator -----------------------------
            self.optimizer_discrim.zero_grad()
            # train on cover
            d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
            d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
            g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

            d_on_cover = self.discriminator(images)
            d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())
            d_loss_on_cover.backward()

            # train on fake
            if self.config.use_mc:
                messages_reduce = self.msg_encoder(messages)
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages_reduce)
            else:
                encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

            d_on_encoded = self.discriminator(encoded_images.detach())
            d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

            d_loss_on_encoded.backward()
            self.optimizer_discrim.step()

            # --------------Train the generator (encoder-decoder) ---------------------
            self.optimizer_enc_dec.zero_grad()
            if self.config.use_mc:
                self.optimizer_msg_encoder.zero_grad()
                self.optimizer_msg_decoder.zero_grad()
            # target label for encoded images should be 'cover', because we want to fool the discriminator
            d_on_encoded_for_enc = self.discriminator(encoded_images)
            g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

            if self.vgg_loss == None:
                g_loss_enc = self.mse_loss(encoded_images, images)
            else:
                vgg_on_cov = self.vgg_loss(images)
                vgg_on_enc = self.vgg_loss(encoded_images)
                g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

# <<<<<<< HEAD
            g_loss_dec_msg = torch.tensor([.0]).to(self.device)
# =======
#             g_loss_dec_msg = torch.tensor([0])
# >>>>>>> 20040681ef050522b5ce942e262ef55a0f48643e
            if self.config.use_mc:
                decoded_messages = self.msg_decoder(decoded_messages)
                g_loss_dec_msg = self.mse_loss(decoded_messages, messages)
                # g_loss_dec_msg = self.msg_coding_loss_weight * self.mse_loss(decoded_messages, messages)
                # g_loss_dec_msg.backward(retain_graph=True)
                # self.optimizer_msg_encoder.step()

            g_loss_dec = self.mse_loss(decoded_messages, messages)
            g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                     + self.config.decoder_loss * g_loss_dec + self.msg_coding_loss_weight * g_loss_dec_msg

            g_loss.backward()
            self.optimizer_enc_dec.step()
            if self.config.use_mc:
                self.optimizer_msg_decoder.step()
                self.optimizer_msg_encoder.step()

        decoded_rounded = utils.clip(decoded_messages.detach().cpu().numpy().round())
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'msg_reduce_mse': g_loss_dec_msg.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def validate_on_batch(self, batch: list):
        """
        Runs validation on a single batch of data consisting of images and messages
        :param batch: batch of validation data, in form [images, messages]
        :return: dictionary of error metrics from Encoder, Decoder, and Discriminator on the current batch
        """
        # if TensorboardX logging is enabled, save some of the tensors.
        if self.tb_logger is not None:
            encoder_final = self.encoder_decoder.encoder._modules['final_layer']
            self.tb_logger.add_tensor('weights/encoder_out', encoder_final.weight)
            decoder_final = self.encoder_decoder.decoder._modules['linear']
            self.tb_logger.add_tensor('weights/decoder_out', decoder_final.weight)
            discrim_final = self.discriminator._modules['linear']
            self.tb_logger.add_tensor('weights/discrim_out', discrim_final.weight)

        images, messages = batch

        batch_size = images.shape[0]

        self.encoder_decoder.eval()
        self.discriminator.eval()

        # It's not necessary to generate the iga mask for the input image during test.
        enable_efficiency = True
        if not enable_efficiency:
            images.requires_grad = True
            if self.config.use_mc:
                messages_enc = self.msg_encoder(messages)
                _, _, decoded_messages = self.encoder_decoder(images, messages_enc)
                deco_msgs = self.msg_decoder(decoded_messages)
            else:
                _, _, deco_msgs = self.encoder_decoder(images, messages)
            mse_loss = self.mse_loss(deco_msgs, messages)
            mse_loss.backward()

            grads = utils.normalize_sigmoid(images.grad)
            images_iga_mask = 1 - grads
            images.grad.zero_()
            images.requires_grad = False

            images.mul_(images_iga_mask)

        d_target_label_cover = torch.full((batch_size, 1), self.cover_label, device=self.device)
        d_target_label_encoded = torch.full((batch_size, 1), self.encoded_label, device=self.device)
        g_target_label_encoded = torch.full((batch_size, 1), self.cover_label, device=self.device)

        d_on_cover = self.discriminator(images)
        d_loss_on_cover = self.bce_with_logits_loss(d_on_cover, d_target_label_cover.float())

        if self.config.use_mc:
            messages_reduce = self.msg_encoder(messages)
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages_reduce)
        else:
            encoded_images, noised_images, decoded_messages = self.encoder_decoder(images, messages)

        d_on_encoded = self.discriminator(encoded_images)
        d_loss_on_encoded = self.bce_with_logits_loss(d_on_encoded, d_target_label_encoded.float())

        d_on_encoded_for_enc = self.discriminator(encoded_images)
        g_loss_adv = self.bce_with_logits_loss(d_on_encoded_for_enc, g_target_label_encoded.float())

        if self.vgg_loss is None:
            g_loss_enc = self.mse_loss(encoded_images, images)
        else:
            vgg_on_cov = self.vgg_loss(images)
            vgg_on_enc = self.vgg_loss(encoded_images)
            g_loss_enc = self.mse_loss(vgg_on_cov, vgg_on_enc)

# <<<<<<< HEAD
        g_loss_dec_msg = torch.tensor([.0]).to(self.device)
# =======
#         g_loss_dec_msg = torch.tensor([0])
# >>>>>>> 20040681ef050522b5ce942e262ef55a0f48643e
        if self.config.use_mc:
            decoded_messages = self.msg_decoder(decoded_messages)
            g_loss_dec_msg = self.msg_coding_loss_weight * self.mse_loss(decoded_messages, messages)

        g_loss_dec = self.mse_loss(decoded_messages, messages)
        g_loss = self.config.adversarial_loss * g_loss_adv + self.config.encoder_loss * g_loss_enc \
                 + self.config.decoder_loss * g_loss_dec + self.msg_coding_loss_weight * g_loss_dec_msg

        decoded_rounded = utils.clip(decoded_messages.detach().cpu().numpy().round())
        bitwise_avg_err = np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                batch_size * messages.shape[1])

        losses = {
            'loss           ': g_loss.item(),
            'encoder_mse    ': g_loss_enc.item(),
            'dec_mse        ': g_loss_dec.item(),
            'bitwise-error  ': bitwise_avg_err,
            'adversarial_bce': g_loss_adv.item(),
            'discr_cover_bce': d_loss_on_cover.item(),
            'discr_encod_bce': d_loss_on_encoded.item(),
            'msg_reduce_mse': g_loss_dec_msg.item()
        }
        return losses, (encoded_images, noised_images, decoded_messages)

    def to_stirng(self):
        return '{}\n{}'.format(str(self.encoder_decoder), str(self.discriminator))
