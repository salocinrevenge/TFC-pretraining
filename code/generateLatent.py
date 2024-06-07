from model import *
import pandas as pd
from config_files.UCI_original_Configs import Config as Configs
import os
import torch
import numpy as np

model_path = "/workspaces/har/salo/TFC-pretraining/code/experiments_logs/UCI_original_2_UCI_original/run1/pre_train_seed_0/saved_models/"
# model_path = "/workspaces/har/salo/TFC-pretraining/code/experiments_logs/UCI_original_2_UCI_original/run1/fine_tune_seed_0/saved_models/"
dataset_path = "/workspaces/har/salo/TFC-pretraining/datasets/UCI_original/test.csv"
device = "cuda"

df = pd.read_csv(dataset_path, header=None)
data = df.values

X = data[:, :-1]
X = X.reshape(X.shape[0], 9, 128)
y = data[:, -1]

X = torch.tensor(X, dtype=torch.float32).to(device)
y = torch.tensor(y, dtype=torch.float32).to(device)


configs = Configs()
model = TFC_model = TFC(configs).to(device)
pretrained_weights = os.path.join(model_path, "ckp_last.pt")
chkpoint = torch.load(pretrained_weights, map_location=device) # two saved models: ['model_state_dict', 'temporal_contr_model_state_dict']


pretrained_dict = chkpoint["model_state_dict"] # Time domain parameters
model_dict = TFC_model.state_dict()
# pretrained_dict = remove_logits(pretrained_dict)
model_dict.update(pretrained_dict)
TFC_model.load_state_dict(model_dict)

model.eval()
# outputs = []
# with torch.no_grad():
#     for i in range(0, len(X)):
#         fft = torch.abs(torch.fft.fft(X[i]))
#         print(fft.shape)
#         h_t, z_t, h_f, z_f = model(X[i], fft)
#         output = torch.concat([z_t, z_f], dim=1)
#         outputs.append([output,y[i]])
# outputs = torch.tensor(outputs)
# print(outputs.shape)

outputs = torch.tensor([]).to(device)
with torch.no_grad():
    fft = torch.fft.fft(X, dim=2)
    fft = torch.abs(fft).to(device)
    h_t, z_t, h_f, z_f = model(X, fft)
    output = torch.concat([z_t, z_f], dim=1)
    # outputs.append([output,y])
    outputs = torch.cat([outputs, output], dim=0)
y = y.reshape(y.shape[0],1)
outputs = torch.cat([outputs, y], dim=1)
outputs = outputs.cpu().numpy()
# salvar vetor em .npy
np.save("latent.npy", outputs)