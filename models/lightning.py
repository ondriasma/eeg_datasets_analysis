"""
models/lightning.py
-------------------
LightningModule wrapper around any Braindecode nn.Module.

Has no imports from the rest of this project so it can
be tested or reused independently.
"""

from __future__ import annotations

import pytorch_lightning as pl
import torch
import torch.nn as nn


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        n_classes: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.01,
    ) -> None:
        super().__init__()
        self.model         = model
        self.n_classes     = n_classes
        self.learning_rate = learning_rate
        self.weight_decay  = weight_decay
        self.loss_fn       = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y  = batch
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        acc   = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        x, y  = batch
        y_hat = self(x)
        loss  = self.loss_fn(y_hat, y)
        acc   = (y_hat.argmax(dim=1) == y).float().mean()
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc',  acc,  on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> dict:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=10, verbose=True,
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val_acc'},
        }
