import datetime
import os.path

from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn

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
        self.betas = kwargs.get('betas', (0.5, 0.999))

        self.max_summary_images: int = kwargs.get('max_summary_images', 4)
        self.snapshot_n: int = kwargs.get('snapshot_n', 50)  # Save model each n epochs

        self.summary_writer = SummaryWriter()

        self.generator = UNet()
        self.discriminator = PatchGAN()

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)

        dataset = PokemonDataset()
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True,
                                     pin_memory=True, num_workers=self.num_workers)
        self.total_iterations = len(dataset) // self.batch_size

        self.criterion = nn.BCEWithLogitsLoss()
        self.criterion_L1 = nn.L1Loss()

        self.epoch = 0

    def train(self):
        self.generator.to(self.device)
        self.discriminator.to(self.device)

        for self.epoch in range(1, self.epochs+1):
            for i, (poke_img_batch, edge_img_batch) in tqdm(enumerate(self.dataloader), total=self.total_iterations,
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

            if self.epoch % self.snapshot_n == 0:
                self.save_model()

    def freeze_discriminators_params(self):
        for param in self.discriminator.parameters():
            param.requires_grad = False

    def unfreeze_discriminators_params(self):
        for param in self.discriminator.parameters():
            param.requires_grad = True

    def save_model(self):
        time_obj = datetime.datetime.now()
        timestamp = time_obj.strftime("%d-%H:%M")
        f = f'model_e{self.epoch}_lr{self.lr}_lambda{self.lambda_l1}_betas{self.betas}_{timestamp}.pth'
        path = os.path.join('pretrained_models', f)

        torch.save({
            'epoch': self.epoch,
            'generator_state_dict': self.generator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, path)
