from torch import nn
import torch
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


class TFC(nn.Module): # Frequency domain encoder
    def __init__(self, configs):
        super(TFC, self).__init__()

        # self.conv_block1_t = nn.Sequential(
        #     nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
        #               stride=configs.stride, bias=False, padding=(configs.kernel_size//2)),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     nn.Dropout(configs.dropout)
        # )

        # self.conv_block2_t = nn.Sequential(
        #     nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # )

        # self.conv_block3_t = nn.Sequential(
        #     nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(configs.final_out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        # )

        enc_t = CNN_HaEtAl_1D_B(input_shape=(1, 6, 300), num_classes=6)
        enc_f = CNN_HaEtAl_1D_B(input_shape=(1, 6, 300), num_classes=6)
        self.encoder_t = enc_t.backbone
        self.encoder_f = enc_f.backbone


        self.projector_t = nn.Sequential(
            nn.Linear(enc_t.fc_input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.projector_f = nn.Sequential(
            nn.Linear(enc_t.fc_input_channels, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )


        # self.conv_block1_f = nn.Sequential(
        #     nn.Conv1d(configs.input_channels, 32, kernel_size=configs.kernel_size,
        #               stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
        #     nn.BatchNorm1d(32),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        #     nn.Dropout(configs.dropout)
        # )

        # self.conv_block2_f = nn.Sequential(
        #     nn.Conv1d(32, 64, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(64),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        # )

        # self.conv_block3_f = nn.Sequential(
        #     nn.Conv1d(64, configs.final_out_channels, kernel_size=8, stride=1, bias=False, padding=4),
        #     nn.BatchNorm1d(configs.final_out_channels),
        #     nn.ReLU(),
        #     nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        # )



    def forward(self, x_in_t, x_in_f):

        """Time-based Contrastive Encoder"""
        # print("entrada:")
        # print(x_in_t.shape)
        # x = self.conv_block1_t(x_in_t)
        # print(x.shape)
        # x = self.conv_block2_t(x)
        # print(x.shape)
        # x = self.conv_block3_t(x)
        # print(x.shape)
        x_in_t = x_in_t.unsqueeze(1)
        x = self.encoder_t(x_in_t)
        h_time = x.reshape(x.shape[0], -1)
        """Cross-space projector"""
        z_time = self.projector_t(h_time)

        """Frequency-based contrastive encoder"""
        # f = self.conv_block1_f(x_in_f)
        # f = self.conv_block2_f(f)
        # f = self.conv_block3_f(f)
        x_in_f = x_in_f.unsqueeze(1)
        f = self.encoder_f(x_in_f)
        h_freq = f.reshape(f.shape[0], -1)

        """Cross-space projector"""
        z_freq = self.projector_f(h_freq)

        return h_time, z_time, h_freq, z_freq


class target_classifier(nn.Module): # Frequency domain encoder
    def __init__(self, configs):
        super(target_classifier, self).__init__()
        self.fc = CNN_HaEtAl_1D_B(input_shape=(1, 6, 300), num_classes=6).fc
        # self.logits = nn.Linear(2*128, 64)
        # self.logits_simple = nn.Linear(64, configs.num_classes_target)

    def forward(self, emb):
        # """2-layer MLP"""
        emb_flat = emb.reshape(emb.shape[0], -1)
        return self.fc(emb_flat)
        # emb = torch.sigmoid(self.fc(emb_flat))
        # pred = self.logits_simple(emb)
        # return emb
