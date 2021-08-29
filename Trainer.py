import datetime
import os.path
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import PokemonDataset
from models.PatchGAN import PatchGAN
from models.Unet import UNet


class Trainer:
    def __init__(self, **kwargs):
        self.lr: int = kwargs.get('lr', 1e-4)
        self.batch_size: int = kwargs.get('batch_size', 64)
        self.lambda_l1: int = kwargs.get('lambda_l1', 100)
        self.epochs: int = kwargs.get('epochs', 120)
        self.num_workers = kwargs.get('dataloader_num_workers', 4)
        self.device: str = kwargs.get('device', 'cpu')
        self.betas: Tuple[float, float] = kwargs.get('betas', (0.5, 0.999))

        self.max_summary_images: int = kwargs.get('max_summary_images', 4)
        self.snapshot_n: int = kwargs.get('snapshot_n', 50)  # Save model each n epochs

        self.summary_writer = SummaryWriter()

        self.generator = UNet()
        self.discriminator = PatchGAN()

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        train_dataset = PokemonDataset(phase='train')
        val_dataset = PokemonDataset(phase='val')
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                           pin_memory=True, num_workers=self.num_workers)
        self.val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True,
                                         pin_memory=True, num_workers=self.num_workers)
        self.total_iterations: int = len(train_dataset) // self.batch_size

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        self.epoch: int = 0

        subfolder_name = datetime.datetime.now().strftime("%d_%H-%M")
        self.output_path: str = os.path.join('trained_models', subfolder_name)
        self.save_param_file()

    def train(self) -> None:
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        for self.epoch in range(1, self.epochs+1):
            for i, (poke_img_batch, edge_img_batch) in tqdm(enumerate(self.train_dataloader),
                                                            total=self.total_iterations,
                                                            desc=f"Epoch: {self.epoch}", unit="batches"):
                global_step = self.epoch * self.total_iterations + i

                poke_img_batch = poke_img_batch.to(self.device)
                edge_img_batch = edge_img_batch.to(self.device)

                fake_image_batch = self.generator(edge_img_batch).to(self.device).detach()

                # Train discriminator

                self.optimizer_d.zero_grad()

                pred_fake = self.discriminator(edge_img_batch, fake_image_batch)
                loss_d_fake = self.criterion(pred_fake, torch.zeros_like(pred_fake).to(self.device))

                pred_real = self.discriminator(edge_img_batch, poke_img_batch)
                loss_d_real = self.criterion(pred_real, torch.ones_like(pred_real).to(self.device))

                loss_d = (loss_d_fake + loss_d_real) * 0.5
                self.summary_writer.add_scalar("Discriminator loss", loss_d, global_step)

                loss_d.backward()
                self.optimizer_d.step()

                # Train generator

                self.freeze_discriminators_params()

                self.optimizer_g.zero_grad()

                gen_imgs = self.generator(edge_img_batch)
                pred_fake = self.discriminator(edge_img_batch, gen_imgs)

                loss_g_GAN = self.criterion(pred_fake, torch.ones_like(pred_fake).to(self.device))
                loss_g_L1 = self.criterion_L1(gen_imgs, poke_img_batch)
                loss_g = loss_g_GAN + self.lambda_l1 * loss_g_L1
                self.summary_writer.add_scalar("Generator loss", loss_g, global_step)

                loss_g.backward()

                self.optimizer_g.step()

                if i % 50 == 0:  # Add images to summary writer every 50 iterations
                    self.summary_writer.add_images("Original images", poke_img_batch[:self.max_summary_images],
                                                   global_step)
                    self.summary_writer.add_images("Generated images", gen_imgs[:self.max_summary_images],
                                                   global_step)

                self.unfreeze_discriminators_params()

            # We don't set the model to eval mode. Same as in the paper.
            cumulative_loss = 0.0
            for i, (poke_img_batch, edge_img_batch) in enumerate(self.train_dataloader):
                with torch.no_grad():  # we don't need to track gradients during evaluation
                    poke_img_batch = poke_img_batch.to(self.device)
                    edge_img_batch = edge_img_batch.to(self.device)

                    gen_imgs = self.generator(edge_img_batch)
                    gen_l1_loss = self.criterion_L1(gen_imgs, poke_img_batch)

                    cumulative_loss += gen_l1_loss

            self.summary_writer.add_scalar("Generator validation L1 loss", cumulative_loss / len(self.train_dataloader),
                                           self.epoch)

            if self.epoch % self.snapshot_n == 0:
                self.save_model()

    def freeze_discriminators_params(self) -> None:
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def unfreeze_discriminators_params(self) -> None:
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def save_param_file(self) -> None:
        """ Param file is used for trained models distinction.  """
        # Create subfolder structure if it doesn't exist
        Path(self.output_path).mkdir(parents=True, exist_ok=True)

        path = os.path.join(self.output_path, 'params.txt')
        with open(path, 'w') as f:
            f.write(f'lr: {self.lr} '
                    f'batch size: {self.batch_size} '
                    f'lambda: {self.lambda_l1} '
                    f'beta1: {self.betas[0]} '
                    f'beta2: {self.betas[1]} '
                    f'max epochs: {self.epochs} ')

    def save_model(self) -> None:
        filename = f'model_epoch_{self.epoch}.pt'
        path = os.path.join(self.output_path, filename)

        torch.save({
            'epoch': self.epoch,
            'lr': self.lr,
            'betas': self.betas,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, path)
