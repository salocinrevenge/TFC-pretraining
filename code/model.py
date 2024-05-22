from torch import nn
import torch
from typing import List, Tuple

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
        self, input_features: int, n_out
    ) -> torch.nn.Module:
        return torch.nn.Sequential(
            torch.nn.Linear(in_features=input_features, out_features=256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(in_features=256, out_features=n_out),
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

        self.n_out = 128*2
        self.n_features = self._calculate_fc_input_features(self.encoder_t, input_shape)
        self.projector_t = self._create_fc(self.n_features, self.n_out//2)
        self.projector_f = self._create_fc(self.n_features, self.n_out//2)



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
