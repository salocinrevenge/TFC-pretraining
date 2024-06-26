import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
import numpy as np
from augmentations import DataTransform_FD, DataTransform_TD
import torch.fft as fft
import pandas as pd

class Load_Dataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, dataset, config, training_mode, target_dataset_size=64, subset=False, percent = 1.0):
        super(Load_Dataset, self).__init__()
        self.training_mode = training_mode
        X_train = dataset["samples"]
        y_train = dataset["labels"]
        # shuffle
        data = list(zip(X_train, y_train))
        np.random.shuffle(data)
        X_train, y_train = zip(*data)
        X_train, y_train = torch.stack(list(X_train), dim=0), torch.stack(list(y_train), dim=0)

        if len(X_train.shape) < 3:
            X_train = X_train.unsqueeze(2)

        if X_train.shape.index(min(X_train.shape)) != 1:  # make sure the Channels in second dim
            X_train = X_train.permute(0, 2, 1)

        """Align the TS length between source and target datasets"""
        X_train = X_train[:, :, :int(config.TSlength_aligned)] # take the first 178 samples

        """Subset for debugging"""
        if subset == True:
            subset_size = target_dataset_size *10
            """if the dimension is larger than 178, take the first 178 dimensions. If multiple channels, take the first channel"""
            X_train = X_train[:subset_size] #
            y_train = y_train[:subset_size]
        if percent < 1.0:
            subset_size = int(X_train.shape[0]*percent)
            X_train = X_train[:subset_size]
            y_train = y_train[:subset_size]

        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train)
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train
            self.y_data = y_train

        """Transfer x_data to Frequency Domain. If use fft.fft, the output has the same shape; if use fft.rfft, 
        the output shape is half of the time window."""

        window_length = self.x_data.shape[-1]
        self.x_data_f = fft.fft(self.x_data).abs() #/(window_length) # rfft for real value inputs.
        # self.x_data_f = self.x_data_f[:, :, 1:] # not a problem.

        self.len = X_train.shape[0]
        """Augmentation"""
        if training_mode == "pre_train":  # no need to apply Augmentations in other modes
            self.aug1 = DataTransform_TD(self.x_data, config)
            self.aug1_f = DataTransform_FD(self.x_data_f, config) # [7360, 1, 90]


    def __getitem__(self, index):
        if self.training_mode == "pre_train":
            return self.x_data[index], self.y_data[index], self.aug1[index],  \
                   self.x_data_f[index], self.aug1_f[index]
        else:
            return self.x_data[index], self.y_data[index], self.x_data[index], \
                   self.x_data_f[index], self.x_data_f[index]

    def __len__(self):
        return self.len

def convert(dataset, ncanais = 6, original = False, tamanho = 60):
    if original:
        dataset = np.asarray(dataset)
        dataset = {"samples": torch.tensor(dataset[:, :tamanho*ncanais],dtype=torch.float64), "labels": torch.tensor(dataset[:,tamanho*ncanais],dtype=torch.int32)}
        dataset["samples"] = dataset["samples"].reshape(dataset["samples"].shape[0], ncanais, -1)
    else:
        dataset = {"samples": torch.tensor(dataset.iloc[:, :tamanho*ncanais].values,dtype=torch.float64), "labels": torch.tensor(dataset.iloc[:,-1].values,dtype=torch.int32)}
        dataset["samples"] = dataset["samples"].reshape(dataset["samples"].shape[0], ncanais, -1)
    return dataset

def data_generator(sourcedata_path, targetdata_path, configs, configs_target, training_mode, subset = True, percent = 1.0):
    csv_s = False
    if "UCI" in sourcedata_path or "KuHar" in sourcedata_path:
        csv_s = True
    original_s = False
    if "original" in sourcedata_path:
        original_s = True
    if csv_s:
        if original_s:
            train_dataset = pd.read_csv(os.path.join(sourcedata_path, "train.csv"), header=None)

            train_dataset = convert(train_dataset, configs.input_channels, True, configs.TSlength_aligned)
        else:
            train_dataset = pd.read_csv(os.path.join(sourcedata_path, "train.csv"))

            train_dataset = convert(train_dataset)
    else:
        train_dataset = torch.load(os.path.join(sourcedata_path, "train.pt"))

    csv_t = False
    if "UCI" in targetdata_path or "KuHar" in targetdata_path:
        csv_t = True
    original_t = False
    if "original" in targetdata_path:
        original_t = True
    if csv_t:
        if original_t:
            finetune_dataset = pd.read_csv(os.path.join(targetdata_path, "train.csv"), header=None)
            test_dataset = pd.read_csv(os.path.join(targetdata_path, "test.csv"), header=None)

            finetune_dataset = convert(finetune_dataset, configs_target.input_channels, True, configs_target.TSlength_aligned)
            test_dataset = convert(test_dataset, configs_target.input_channels, True, configs_target.TSlength_aligned)
        else:
            finetune_dataset = pd.read_csv(os.path.join(targetdata_path, "train.csv"))
            test_dataset = pd.read_csv(os.path.join(targetdata_path, "test.csv"))

            finetune_dataset = convert(finetune_dataset)
            test_dataset = convert(test_dataset)
    else:
        finetune_dataset = torch.load(os.path.join(targetdata_path, "train.pt"))
        test_dataset = torch.load(os.path.join(targetdata_path, "test.pt"))
    """ Dataset notes:
    Epilepsy: train_dataset['samples'].shape = torch.Size([7360, 1, 178]); binary labels [7360] 
    valid: [1840, 1, 178]
    test: [2300, 1, 178]. In test set, 1835 are positive sampels, the positive rate is 0.7978"""
    """sleepEDF: finetune_dataset['samples']: [7786, 1, 3000]"""

    # subset = True # if true, use a subset for debugging.
    train_dataset = Load_Dataset(train_dataset, configs, training_mode, target_dataset_size=configs.batch_size, subset=subset, percent=percent) # for self-supervised, the data are augmented here
    finetune_dataset = Load_Dataset(finetune_dataset, configs_target, training_mode, target_dataset_size=configs.target_batch_size, subset=subset, percent=percent)
    # if test_dataset['labels'].shape[0]>10*configs.target_batch_size:
    #     test_dataset = Load_Dataset(test_dataset, configs_target, training_mode, target_dataset_size=configs.target_batch_size*10, subset=subset, percent=1.0)
    # else:
    #     test_dataset = Load_Dataset(test_dataset, configs_target, training_mode, target_dataset_size=configs.target_batch_size, subset=subset, percent=1.0)
    test_dataset = Load_Dataset(test_dataset, configs_target, training_mode, target_dataset_size=len(test_dataset), subset=False, percent=1.0)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=configs.batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    """the valid and test loader would be finetuning set and test set."""
    valid_loader = torch.utils.data.DataLoader(dataset=finetune_dataset, batch_size=configs.target_batch_size,
                                               shuffle=True, drop_last=configs.drop_last,
                                               num_workers=0)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=configs.target_batch_size,
                                              shuffle=True, drop_last=False,
                                              num_workers=0)

    return train_loader, valid_loader, test_loader