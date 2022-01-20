import argparse

import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from dataset import PdfPageDataModule
from model import ArcMargineLoss, FeatureExtractor


class LitImageSearchModel(pl.LightningModule):
    def __init__(
        self, num_classes: int, dim: int, binarize: bool, margin: float, scale: float
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = FeatureExtractor(dim, binarize)
        self.loss_func = ArcMargineLoss(num_classes, dim, margin, scale)

    def training_step(self, batch, batch_idx):
        x, y = batch
        feature = self.model(x)
        loss = self.loss_func(feature, y)
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Garellyを登録する
        """
        self.model.eval()
        val_garelly = []
        with torch.no_grad():
            for x, _ in self.trainer.datamodule.test_dataloader():
                x = x.to(self.device)
                val_garelly.append(self.model(x))
        val_garelly = torch.cat(val_garelly, dim=0)  # [n, d]
        self.val_garelly = F.normalize(val_garelly).detach()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)  # [N, d]
        out = F.normalize(out)

        cos = torch.einsum("Nd,nd->Nn", out, self.val_garelly)  # [N, n]
        pred = torch.argmax(cos, dim=1)  # [N,]
        acc = (pred == y).sum().item() / y.shape[0]

        self.log("val_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {"params": self.model.parameters(), "lr": 1e-5},
                {"params": self.loss_func.parameters()},
            ],
            lr=1e-3,
            weight_decay=1e-4,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("LitImageSearchModel")
        parser.add_argument(
            "--dim", type=int, default=256, help="the dimension of the feature."
        )
        parser.add_argument(
            "--binarize", action="store_true", help="binarize feature or not."
        )
        parser.add_argument(
            "--margin", type=float, default=0.5, help="margin of ArcMargineLoss."
        )
        parser.add_argument(
            "--scale", type=float, default=10.0, help="scale of ArcMargineLoss."
        )
        return parent_parser


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset_dir", type=str, help="path to image directory.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")

    parser = LitImageSearchModel.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)

    args = parser.parse_args()

    datamodule = PdfPageDataModule(args.dataset_dir, args.batch_size)
    datamodule.setup("fit")

    litmodule = LitImageSearchModel(
        len(datamodule.train_dataset), args.dim, args.binarize, args.margin, args.scale
    )
    trainer = pl.Trainer.from_argparse_args(args, check_val_every_n_epoch=5)
    trainer.fit(litmodule, datamodule)

    script = torch.jit.script(litmodule.model)
    script.save("feature.pt")


if __name__ == "__main__":
    main()
