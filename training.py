import PIL
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from models.Unet import UNet
from models.PatchGAN import PatchGAN
from dataset import PokemonDataset

IMG_SIZE = 256
CHANNELS = 3

MAX_SUMMARY_IMAGES = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CUDA = torch.cuda.is_available()

LR = 2e-4
EPOCHS = 60
BATCH_SIZE = 64
NUM_WORKERS = 4
CLIP_VALUE = 1e-1
LAMBDA_L1 = 200


def train():
    summary_writer = SummaryWriter()

    generator = UNet()
    critic = PatchGAN()

    critic.to(DEVICE)
    generator.to(DEVICE)

    dataset = PokemonDataset()
    total_iterations = len(dataset) // BATCH_SIZE
    data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=NUM_WORKERS)

    optimizer_c = torch.optim.Adam(critic.parameters(), lr=LR, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=LR, betas=(0.5, 0.999))

    criterion = nn.BCEWithLogitsLoss()
    criterion_L1 = nn.L1Loss()

    for epoch in range(1, EPOCHS):
        if epoch < 10:
            c_times = 100
        else:
            c_times = 5

        for i, (poke_img_batch, edge_img_batch) in tqdm(enumerate(data_loader), total=total_iterations, desc=f"Epoch: {epoch}", unit="batches"):
            global_step = epoch * total_iterations + i

            poke_img_batch = poke_img_batch.to(DEVICE)
            edge_img_batch = edge_img_batch.to(DEVICE)
            fake_image_batch = generator(edge_img_batch).to(DEVICE).detach()

            optimizer_c.zero_grad()

            pred_fake = critic(edge_img_batch, fake_image_batch)
            loss_c_fake = criterion(pred_fake, torch.zeros(pred_fake.size()).to(DEVICE))

            pred_real = critic(edge_img_batch, poke_img_batch)
            loss_c_real = criterion(pred_real, torch.ones(pred_real.size()).to(DEVICE))

            loss_c = (loss_c_fake + loss_c_real) * 0.5
            summary_writer.add_scalar("Critic loss", loss_c, global_step)

            loss_c.backward()

            optimizer_c.step()

            for p in critic.parameters():
                p.data.clamp_(-CLIP_VALUE, CLIP_VALUE)

            if i % c_times == 0:
                optimizer_g.zero_grad()

                gen_imgs = generator(edge_img_batch)
                pred_fake = critic(edge_img_batch, gen_imgs)

                loss_g_GAN = criterion(pred_fake, torch.ones(pred_fake.size()).to(DEVICE))
                loss_g_L1 = criterion_L1(gen_imgs, poke_img_batch)
                loss_g = loss_g_GAN + LAMBDA_L1 * loss_g_L1

                loss_g.backward()

                optimizer_g.step()

                # summary_writer.add_image("Generated images", gen_imgs[0], global_step)
                summary_writer.add_images("Generated images", gen_imgs[:MAX_SUMMARY_IMAGES], global_step)
                summary_writer.add_scalar("Generator loss", loss_g, global_step)


def tensor_to_image(tensor):

    gen_img = tensor
    tensor_image = gen_img.view(gen_img.shape[1], gen_img.shape[2], gen_img.shape[0])
    print(tensor_image)
    tensor_image = tensor_image.detach().to('cpu').numpy()

    print(type(tensor_image), tensor_image.shape)

    plt.imshow(tensor_image)
    plt.show()


if __name__ == "__main__":
    train()