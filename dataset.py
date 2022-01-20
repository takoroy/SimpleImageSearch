import glob
import itertools
import os
import re
from collections import defaultdict
from typing import Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import (ColorJitter, Compose, GaussianBlur,
                                    Normalize, RandomApply, RandomPerspective,
                                    RandomResizedCrop, Resize, ToTensor)


class PdfPageDataset(Dataset):
    def __init__(self, root_dir: str, split: str, augmentation: bool = True) -> None:
        re_fname = re.compile(r"(\S{5})_(\d{4})\.jpg$")

        pages_by_book = defaultdict(list)
        for fpath in glob.glob(os.path.join(root_dir, "*.jpg")):
            book_id, _ = re_fname.search(fpath).groups()
            pages_by_book[book_id].append(fpath)

        books = sorted(pages_by_book.keys())
        if split == "train":
            books = books[:15]
        elif split == "val":
            books = books[15:20]
        elif split == "predict":
            books = books
        else:
            raise Exception(f"Split {split} is not defined.")

        if split == "train" or augmentation:
            self.transform = Compose(
                [
                    RandomPerspective(),
                    ColorJitter(brightness=.5, hue=.3),
                    RandomResizedCrop(320, scale=(0.2, 1)),
                    RandomApply([GaussianBlur(5), GaussianBlur(11)], p=0.5),
                    ToTensor(),
                    Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )
        else:
            self.transform = Compose(
                [Resize(480), ToTensor(), Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),]
            )

        self.data = sorted(
            itertools.chain.from_iterable([pages_by_book[book_id] for book_id in books])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.LongTensor]:
        fpath = self.data[index]
        img = Image.open(fpath).convert("RGB")
        img = self.transform(img)
        return img, index


class PdfPageDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        if stage == "fit":
            self.train_dataset = PdfPageDataset(self.data_dir, "train")
            self.val_dataset = PdfPageDataset(self.data_dir, "val")
            self.test_dataset = PdfPageDataset(self.data_dir, "val", augmentation=False)
        else:
            self.predict_gallery = PdfPageDataset(
                self.data_dir, "predict", augmentation=False
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True,
            shuffle=True,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)

    def predict_dataloader(self):
        return DataLoader(self.predict_gallery, batch_size=1)
