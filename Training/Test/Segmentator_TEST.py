import os
import numpy as np
import pandas as pd
from Metricas.Metrics import *
import pytorch_lightning as pl


class Segmentor_TEST_mini(pl.LightningModule):
    def __init__(self, model, model_type, save_dir=None):
        super().__init__()
        self.model = model
        self.model_type = model_type
        self.save_dir = save_dir

        # Lista para métricas por imagen
        self.test_outputs = []

    def forward(self, x):
        return self.model(x)

    def test_step(self, batch, batch_idx):
        image = batch['image']
        y_true = batch['mask']
        img_name = batch['name']  

        y_pred = self.model(image)

        dice_score = dice_hard(y_true, y_pred)
        iou_score = iou(y_true, y_pred)
        presicion_score = precision(y_true, y_pred)
        recall_score = recall(y_true, y_pred)
        ASSD_score = ASSD(y_true, y_pred)
        Boundaryiou_score = boundary_iou(y_true, y_pred)

        metrics = {
            "image_name": img_name[0],  # batch_size=1
            "dice": dice_score.detach().cpu().item(),
            "iou": iou_score,
            "precision": presicion_score,
            "recall": recall_score.detach().cpu().item(),
            "assd": ASSD_score.detach().cpu().item(),
            "boundaryiou": Boundaryiou_score.detach().cpu().item(),
        }

        self.test_outputs.append(metrics)
        return metrics

    def on_test_epoch_end(self):

        # -------- STACK --------
        dice = np.array([([x["dice"] for x in self.test_outputs])])
        iou = np.array([([x["iou"] for x in self.test_outputs])])
        precision = np.array([([x["precision"] for x in self.test_outputs])])
        recall = np.array([([x["recall"] for x in self.test_outputs])]))
        assd = np.array([([x["assd"] for x in self.test_outputs])])
        boundaryiou = np.array([([x["boundaryiou"] for x in self.test_outputs])])

        # -------- MEAN --------
        dice_mean = dice.mean()
        iou_mean = iou.mean()
        precision_mean = precision.mean()
        recall_mean = recall.mean()
        assd_mean = assd.mean()
        boundaryiou_mean = boundaryiou.mean()

        # -------- STD --------
        dice_std = dice.std()
        iou_std = iou.std()
        precision_std = precision.std()
        recall_std = recall.std()
        assd_std = assd.std()
        boundaryiou_std = boundaryiou.std()

        # -------- LOG (mantienes tu comportamiento actual) --------

        self.log("Test Dice Mean", dice_mean)
        self.log("Test Dice Std", dice_std)

        self.log("Test Iou Mean", iou_mean)
        self.log("Test Iou Std", iou_std)

        self.log("Test Precision Mean", precision_mean)
        self.log("Test Precision Std", precision_std)

        self.log("Test Recall Mean", recall_mean)
        self.log("Test Recall Std", recall_std)


        self.log("Test ASSD Mean", assd_mean)
        self.log("Test ASSD Std", assd_std)

        self.log("Test BoundaryIoU Mean", boundaryiou_mean)
        self.log("Test BoundaryIoU Std", boundaryiou_std)

        # -------- CSV POR IMAGEN --------
        if self.save_dir is not None:
            df = pd.DataFrame(self.test_outputs)
            csv_path = os.path.join(self.save_dir, "metricas_por_imagen.csv")
            df.to_csv(csv_path, index=False)

        # Limpiar memoria
        self.test_outputs.clear()