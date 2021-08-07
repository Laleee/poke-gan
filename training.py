import torch

from Trainer import Trainer

MAX_SUMMARY_IMAGES = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LR = 2e-4
EPOCHS = 120
BATCH_SIZE = 64
NUM_WORKERS = 4
LAMBDA_L1 = 100
g

if __name__ == "__main__":
    trainer = Trainer(
        lr=LR,
        device=DEVICE,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        lambda_l1=LAMBDA_L1,
        dataloader_num_workers=NUM_WORKERS,
        max_summary_images=MAX_SUMMARY_IMAGES
    )
    trainer.train()
