from torch import nn
import torch
from typing import List, Tuple
from ssl_tools.models.nets.cnn_ha_etal import CNN_HaEtAl_1D

class CNN_HaEtAl_1D_B(CNN_HaEtAl_1D):
    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )


class CNN_HaEtAl_1D(SimpleClassificationNet):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (1, 6, 60),
        num_classes: int = 6,
        learning_rate: float = 1e-3,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes

        backbone = self._create_backbone(input_shape=input_shape)
        self.fc_input_channels = self._calculate_fc_input_features(
            backbone, input_shape
        )
        fc = self._create_fc(self.fc_input_channels, num_classes)
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

    def _create_backbone(self, input_shape: Tuple[int, int]) -> torch.nn.Module:
        return torch.nn.Sequential(
            # Add padding
            # ZeroPadder2D(
            #     pad_at=self.pad_at,
            #     padding_size=4 - 1,  # kernel size - 1
            # ),
            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(1, 4),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
            
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
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
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
            # torch.nn.Softmax(dim=1),
        )

class TFC(nn.Module): # Frequency domain encoder

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
        random_input = torch.randn(1, *input_shape)
        with torch.no_grad():
            out = backbone(random_input)
        return out.view(out.size(0), -1).size(1)

    def _create_fc(
        self, input_features: int, num_classes: int
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=128, out_features=num_classes),
        )
    
    def __init__(self, input_shape: Tuple[int, int, int] = (1, 6, 300), num_classes: int = 18):
        super(TFC, self).__init__()

        # First 2D convolutional layer
        self.encoder_t = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(1, 4),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
            
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
        )
        self.encoder_f = torch.nn.Sequential(

            # First 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=input_shape[0],
                out_channels=32,
                kernel_size=(1, 4),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
            
            # Second 2D convolutional layer
            torch.nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=(1, 5),
                stride=(1, 1),
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=(1, 3),
                stride=(1, 3),
            ),
        )


        self.projector_t = self._create_fc(self._calculate_fc_input_features(self.encoder_t, input_shape), num_classes)
        self.projector_f = self._create_fc(self._calculate_fc_input_features(self.encoder_f, input_shape), num_classes)



    def forward(self, x_in_t, x_in_f):

        """Time-based Contrastive Encoder"""
        x_in_t = x_in_t.unsqueeze(1)
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        x_in_f = x_in_f.unsqueeze(1)
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq
# class TFC(nn.Module): # Frequency domain encoder
#     def __init__(self, configs):
#         super(TFC, self).__init__()


#         enc_t = CNN_HaEtAl_1D_B(input_shape=(1, 6, 300), num_classes=6)
#         enc_f = CNN_HaEtAl_1D_B(input_shape=(1, 6, 300), num_classes=6)
#         self.encoder_t = enc_t.backbone
#         self.encoder_f = enc_f.backbone


#         self.projector_t = nn.Sequential(
#             nn.Linear(enc_t.fc_input_channels, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )

#         self.projector_f = nn.Sequential(
#             nn.Linear(enc_t.fc_input_channels, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(),
#             nn.Linear(256, 128)
#         )


#         # self.conv_block1_f = nn.Sequential(
#         #     nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
#         #               stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
#         #     nn.BatchNorm1d(32),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         #     nn.Dropout(configs.dropout)
#         # )

#         # self.conv_block2_f = nn.Sequential(
#         #     nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
#         #     nn.BatchNorm1d(64),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
#         # )

#         # self.conv_block3_f = nn.Sequential(
#         #     nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
#         #     nn.BatchNorm1d(configs.final_out_channels),
#         #     nn.ReLU(),
#         #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
#         # )



#     def forward(self, x_in_t, x_in_f):

#         """Time-based Contrastive Encoder"""
#         # print("entrada:")
#         # print(x_in_t.shape)
#         # x = self.conv_block1_t(x_in_t)
#         # print(x.shape)
#         # x = self.conv_block2_t(x)
#         # print(x.shape)
#         # x = self.conv_block3_t(x)
#         # print(x.shape)
#         x_in_t = x_in_t.unsqueeze(1)
#         x = self.encoder_t(x_in_t)
#         h_time = x.reshape(x.shape[0], -1)
#         """Cross-space projector"""
#         z_time = self.projector_t(h_time)

#         """Frequency-based contrastive encoder"""
#         # f = self.conv_block1_f(x_in_f)
#         # f = self.conv_block2_f(f)
#         # f = self.conv_block3_f(f)
#         x_in_f = x_in_f.unsqueeze(1)
#         f = self.encoder_f(x_in_f)
#         h_freq = f.reshape(f.shape[0], -1)

#         """Cross-space projector"""
#         z_freq = self.projector_f(h_freq)

#         return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module): # Frequency domain encoder
    def __init__(self, input_features: int, num_classes: int = 18):
        super(target_classifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, emb):
        # """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        return self.fc(emb_flat)
