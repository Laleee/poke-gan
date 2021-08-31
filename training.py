import torch
import wandb

from Trainer import Trainer

MAX_SUMMARY_IMAGES = 4

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert torch.cuda.is_available()

# LR = 2e-4
EPOCHS = 100
# BATCH_SIZE = 64
NUM_WORKERS = 4
# LAMBDA_L1 = 100

sweep_config = {
    'method': 'bayes',  # grid, random
    'metric': {
        'name': 'loss_g',
        'goal': 'minimize'
    },
    'parameters': {
        'lambda_l1': {
            'values': [80, 90, 100, 110, 120, 130]
        },
        'batch_size': {
            'values': [64]
        },
        'learning_rate': {
            'values': [1e-5, 1e-4, 2e-4, 3e-4]
        }
    }
}

if __name__ == '__main__':
    def train_wrapper():
        wandb.init()
        config = wandb.config
        print(f'Config: {config}')

        trainer = Trainer(
            lr=config.learning_rate,
            device=DEVICE,
            batch_size=config.batch_size,
            epochs=EPOCHS,
            lambda_l1=config.learning_rate,
            dataloader_num_workers=NUM_WORKERS,
            max_summary_images=MAX_SUMMARY_IMAGES
        )
        trainer.train()

    sweep_id = wandb.sweep(sweep_config, project="poke-gan")
    wandb.agent(sweep_id, train_wrapper)
