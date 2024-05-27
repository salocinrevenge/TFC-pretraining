from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import pandas as pd
import lightning as L
import torchmetrics
import numpy as np
from typing import Tuple, Dict
import torch
from torchmetrics import Accuracy
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from torch import nn

class SimpleDataset:
    def __init__(self, X, y, percentage = 1.0):
        self.X = X
        self.y = y
        if percentage < 1.0:
            self.X = self.X[:int(len(self.X)*percentage)]
            self.y = self.y[:int(len(self.y)*percentage)]
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
    
class SimpleDataModule(L.LightningDataModule):
    def __init__(self, train_dset, val_dset, test_dset, batch_size=32):
        super().__init__()
        self.train_dset = train_dset
        self.val_dset = val_dset
        self.test_dset = test_dset
        self.batch_size = batch_size
        
    def train_dataloader(self):
        return DataLoader(self.train_dset, batch_size=self.batch_size, shuffle=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dset, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.test_dset, batch_size=self.batch_size)
    

class SimpleClassificationNet(L.LightningModule):
    def __init__(
        self,
        backbone: torch.nn.Module,
        fc: torch.nn.Module,
        learning_rate: float = 1e-3,
        flatten: bool = True,
        loss_fn: torch.nn.Module = None,
        train_metrics: Dict[str, torch.Tensor] = None,
        val_metrics: Dict[str, torch.Tensor] = None,
        test_metrics: Dict[str, torch.Tensor] = None,
    ):
        super().__init__()
        self.backbone = backbone
        self.fc = fc
        self.learning_rate = learning_rate
        self.flatten = flatten
        self.loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self.metrics = {
            "train": train_metrics or {},
            "val": val_metrics or {},
            "test": test_metrics or {},
        }

    def loss_func(self, y_hat, y):
        loss = self.loss_fn(y_hat, y)
        return loss

    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        if self.flatten:
            x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

    def compute_metrics(self, y_hat, y, step_name):
        for metric_name, metric_fn in self.metrics[step_name].items():
            metric = metric_fn.to(self.device)(y_hat, y)
            self.log(
                f"{step_name}_{metric_name}",
                metric,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

    def single_step(self, batch: torch.Tensor, batch_idx: int, step_name: str):
        x, y = batch
        y_hat = self.forward(x)
        loss = self.loss_func(y_hat, y)
        self.log(
            f"{step_name}_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.compute_metrics(y_hat, y, step_name)
        return loss

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "train")

    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "val")

    def test_step(self, batch: torch.Tensor, batch_idx: int):
        return self.single_step(batch, batch_idx, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, y = batch
        y_hat = self.forward(x)
        return y_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        return optimizer
    
class CNN_TFC(SimpleClassificationNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone1 = self._create_backbone_1()
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone1, input_shape
        )
        backbone2 = self._create_backbone_2(self.fc_input_channels)
        backbone = nn.Sequential(backbone1, reshape_layer(), backbone2)
        fc = self._create_fc(128, num_classes)
        super().__init__(
            backbone=backbone,
            fc=fc,
            learning_rate=learning_rate,
            flatten=True,
            loss_fn=torch.nn.CrossEntropyLoss(),
            val_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
            test_metrics={
                "acc": Accuracy(task="multiclass", num_classes=num_classes)
            },
        )

    def _create_backbone_1(self) -> torch.nn.Module:
        return nn.Sequential(
            # print_layer(0),
            nn.Conv1d(9, 32, kernel_size=8,
                      stride=1, bias=False, padding=(4)),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(0.35),
            # print_layer(1),

            nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # print_layer(2),

            nn.Conv1d(64, 60, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            # print_layer(3),
        )
    
    

    def _create_backbone_2(self, n_features) -> torch.nn.Module:
        return nn.Sequential(
            # print_layer(4),
            nn.Linear(n_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            # print_layer(5),
            
            
            )
                             

    def _calculate_fc_input_features(
        self, backbone: torch.nn.Module, input_shape: Tuple[int, int, int]
    ) -> int:
        """Run a single forward pass with a random input to get the number of
        features after the convolutional layers.

        Parameters
        ----------
        backbone : torch.nn.Module
            The backbone of the network
        input_shape : Tuple[int, int, int]
            The input shape of the network.

        Returns
        -------
        int
            The number of features after the convolutional layers.
        """
        random_input = torch.randn(*input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(128, 64),
            sigmoid_layer(),
            nn.Linear(64, num_classes)

        )
    
    def forward(self, x: torch.Tensor):
        x = self.backbone(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)
        return x
    
class print_layer(nn.Module):
    def __init__(self, i):
        super(print_layer, self).__init__()
        self.i = i

    def forward(self, x):
        print(f'Layer {self.i}: {x.shape}')
        return x

class sigmoid_layer(nn.Module):
    def __init__(self):
        super(sigmoid_layer, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)
    
class reshape_layer(nn.Module):
    def __init__(self):
        super(reshape_layer, self).__init__()

    def forward(self, x):
        return x.reshape(x.shape[0], -1)

csv = True

if csv:
    train = pd.read_csv('../../../datasets/UCI_original/train.csv', header=None)
    validation = pd.read_csv('../../../datasets/UCI_original/train.csv', header=None)
    test = pd.read_csv('../../../datasets/UCI_original/test.csv', header=None)
    X_train, y_train = train.iloc[:, :-1].values, train.iloc[:, -1].values
    X_validation, y_validation = validation.iloc[:, :-1].values, validation.iloc[:, -1].values
    X_test, y_test = test.iloc[:, :-1].values, test.iloc[:, -1].values
else:
    train = torch.load('../../../datasets/Gesture/train.pt')
    validation = torch.load('../../../datasets/Gesture/train.pt')
    test = torch.load('../../../datasets/Gesture/test.pt')

    X_train = train["samples"]
    y_train = train["labels"]
    X_validation = validation["samples"]
    y_validation = validation["labels"]
    X_test = test["samples"]
    y_test = test["labels"]

if csv:
    formato = (-1, 9, 128)
else:
    formato = (-1, 3, 206)
X_train = X_train.reshape(*formato).astype(np.float32)
X_validation = X_validation.reshape(*formato).astype(np.float32)
X_test = X_test.reshape(*formato).astype(np.float32)

y_train = y_train.astype(int)
y_validation = y_validation.astype(int)
y_test = y_test.astype(int)




if csv:
    train = SimpleDataset(X_train, y_train, percentage=1.0)
    validation = SimpleDataset(X_validation, y_validation, percentage=1.0)
    test = SimpleDataset(X_test, y_test, percentage=1.0)

    dm = SimpleDataModule(train, validation, test, batch_size=32)
else:
    
    dm = SimpleDataModule(train, validation, test, batch_size=32)





model = CNN_TFC(input_shape=(1, 9, 128), num_classes=6)









trainer = L.Trainer(max_epochs=40, accelerator="gpu", devices=1)
trainer.fit(model, dm)





import torch
y_hat = trainer.predict(model, dm)
y_hat = torch.cat(y_hat)
y_hat_maxed = torch.argmax(y_hat, dim=1)





print(confusion_matrix(y_test, y_hat_maxed))
print(y_test.shape, y_hat_maxed.shape)



accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=6)
accuracy_score = accuracy_metric(y_hat_maxed, torch.tensor(y_test))
print(accuracy_score.item())

auroc_metric = torchmetrics.AUROC(task="multiclass", num_classes=6)
auroc = auroc_metric(y_hat, torch.tensor(y_test))
print("auroc: ", auroc.item())

auprc_metric = torchmetrics.AveragePrecision(task="multiclass", num_classes=6)
auprc = auprc_metric(y_hat, torch.tensor(y_test))
print("auprc", auprc.item())