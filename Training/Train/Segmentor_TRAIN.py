import torch
import pytorch_lightning as pl
from Datasets.Metrics import *

# Lightning module
import timm.optim

class Segmentor_TRAIN(pl.LightningModule):
    def __init__(self, model, model_type, loss_function, loss_function_name, learning_rate, monitor, mode, batch_size):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.criterion = loss_function
        self.loss_function_name = loss_function_name
        self.learning_rate = learning_rate
        self.batch_size = batch_size

        #Schedular
        self.monitor = monitor
        self.mode = mode

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        image, y_true = batch['image'], batch['mask']
        # Pred
        y_pred = self.model(image)
        # Loss
        loss = self.criterion(y_pred, y_true)

        with torch.no_grad():
            dice_score = dice_hard(y_true, y_pred)

        return loss, dice_score
    
    def training_step(self, batch, batch_idx):
        loss, dice_score = self._step(batch)
        metrics = {"loss": loss, "train_dice": dice_score}
        self.log_dict(metrics, on_step=True, on_epoch=True, prog_bar = True, batch_size=self.batch_size)
        return loss
    
    def on_train_epoch_end(self):
        lr = self.optimizers().param_groups[0]["lr"]
        self.log("lr", lr, prog_bar=True, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        loss, dice_score = self._step(batch)
        metrics = {"val_loss": loss, "val_dice": dice_score}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, batch_size=1)
        return metrics

    def test_step(self, batch, batch_idx):
        loss, dice_score = self._step(batch)
        metrics = {"test_loss": loss, "test_dice": dice_score}
        self.log_dict(metrics, on_epoch=True, prog_bar=True, batch_size=self.batch_size)
        return metrics

    def configure_optimizers(self):
        optimizer = timm.optim.NAdam(self.parameters(),  lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.mode, factor = 0.5, patience=10)
        return {"optimizer": optimizer,"lr_scheduler": {"scheduler": scheduler,"monitor": self.monitor,},}