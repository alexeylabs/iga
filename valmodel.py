import argparse
import os

import torch
import numpy as np
from tqdm import tqdm
import utils
import logging

from noise_argparser import NoiseArgParser
from options import IGAConfiguration, TrainingOptions
from model.iga import IGA
from noise_layers.noiser import Noiser
from average_meter import AverageMeter


def main():
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    parser = argparse.ArgumentParser(description='Validating of IGA nets')
    parser.add_argument('--size', '-s', default=128, type=int, help='The size of the images (images are square so this is height and width).')
    parser.add_argument('--noise', nargs='*', action=NoiseArgParser,
                        help="Noise layers configuration. Use quotes when specifying configuration, e.g. 'cropout((0.55, 0.6), (0.55, 0.6))'")
    parser.add_argument('--data-dir', '-d', required=True, type=str, help='The directory where the data is stored.')
    parser.add_argument('--model-path', '-p', required=True, type=str, help='The directory where the images is stored.')
    parser.add_argument('--message', '-m', default=30, type=int, help='The length in bits of the watermark.')
    args = parser.parse_args()

    noise_config = args.noise
    iga_config = IGAConfiguration(H=args.size, W=args.size,
                                  message_length=args.message,
                                  message_middle_length=30,
                                  encoder_blocks=4, encoder_channels=64,
                                  decoder_blocks=7, decoder_channels=64,
                                  use_discriminator=True,
                                  use_vgg=False,
                                  use_mc=True,
                                  discriminator_blocks=3, discriminator_channels=64,
                                  decoder_loss=1,
                                  encoder_loss=0.7,
                                  adversarial_loss=1e-3,
                                  enable_fp16=False
                                  )

    train_options = TrainingOptions(
        batch_size=32,
        number_of_epochs=15,
        train_folder=os.path.join(args.data_dir, 'train'),
        validation_folder=os.path.join(args.data_dir, 'val'),
        runs_folder=os.path.join('.', 'runs'),
        start_epoch=100,
        experiment_name='test')

    noiser = Noiser(noise_config, device)
    model = IGA(iga_config, device, noiser, tb_logger=None)
    checkpoint = torch.load(args.model_path, map_location=device)
    utils.model_from_checkpoint(model, checkpoint)
    augmentations = model.encoder_decoder.noiser.noise_layers

    _, val_data = utils.get_data_loaders(iga_config, train_options)

    for aug in augmentations:
        print(aug)
        model.encoder_decoder.noiser.noise_layers = [aug]

        first_iteration = True
        validation_losses = {}
        for image, _ in tqdm(val_data):
            image = image.to(device)
            message = torch.Tensor(np.random.choice([0, 1], (image.shape[0], iga_config.message_length))).to(device)
            losses, (encoded_images, noised_images, decoded_messages) = model.validate_on_batch([image, message])
            if not validation_losses:  # dict is empty, initialize
                for name in losses:
                    validation_losses[name] = AverageMeter()
            for name, loss in losses.items():
                validation_losses[name].update(loss)

        utils.print_progress(validation_losses)


if __name__ == '__main__':
    main()
