import os

import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class PokemonDataset(Dataset):
    def __init__(self, image_size: int = 256, phase: str = 'train'):
        if phase not in ['train', 'val']:
            raise ValueError(f'Phase "{phase}" not supported. Currently supporting only "train" and "val".')

        self.image_size: int = image_size
        dataset_root_path: str = os.path.join('data', phase)
        self.dataset_root_path = os.path.abspath(dataset_root_path)

        self.pokemons_root_path = os.path.join(self.dataset_root_path, 'pokemon_jpg')
        self.pokemons_paths = sorted(
            [os.path.join(self.pokemons_root_path, f) for f in os.listdir(self.pokemons_root_path)
             if f.endswith('.jpg')])

        self.edges_root_path = os.path.join(self.dataset_root_path, 'sketch_jpg')
        self.edges_paths = sorted([os.path.join(self.edges_root_path, f) for f in os.listdir(self.edges_root_path)
                                   if f.endswith('.jpg')])

        self.pokemon_transforms = transforms.Compose([
            transforms.ToTensor()
        ])
        self.edge_transforms = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
        ])

        assert len(self.pokemons_paths) == len(self.edges_paths) > 0, 'Pokemon size and edge size differ'

    def __len__(self):
        return len(self.pokemons_paths)

    def __getitem__(self, item):
        image_path = self.pokemons_paths[item]
        edge_path = self.edges_paths[item]
        image = Image.open(image_path)
        edge = Image.open(edge_path)

        return self.pokemon_transforms(image), self.edge_transforms(edge)
