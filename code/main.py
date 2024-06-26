import torch

import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger, set_requires_grad
#from model import TFC
from utils import _calc_metrics, copy_Files
from model import * # base_Model, base_Model_F, target_classifier

from dataloader import data_generator
from trainer import Trainer, model_finetune, model_test #model_evaluate


# Args selections
start_time = datetime.now()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
# parser.add_argument('--experiment_description', default='Exp1', type=str,
#                     help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int, help='seed value')

# 1. self_supervised; 2. finetune (itself contains finetune and test)
parser.add_argument('--training_mode', default='fine_tune_test', type=str,
                    help='pre_train, fine_tune_test')

parser.add_argument('--pretrain_dataset', default='SleepEEG', type=str,
                    help='Dataset of choice: SleepEEG, FD_A, HAR, ECG')
parser.add_argument('--target_dataset', default='Epilepsy', type=str,
                    help='Dataset of choice: Epilepsy, FD_B, Gesture, EMG')

parser.add_argument('--logs_save_dir', default='experiments_logs', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--percent', default=1.0, type=float,
                    help='percent of data used in training')

parser.add_argument('--epochs', default=40, type=int,
                    help='number of epochs in training')

parser.add_argument('--batch', default=None, type=int,
                    help='number of batchs in training')
parser.add_argument('--single_pipe', default=None, type=str,
                    help='what pipeline the model should use. time or frequency or None for both')

parser.add_argument('--home_path', default=home_dir, type=str,
                    help='Project home directory')
# args = parser.parse_args()
args, unknown = parser.parse_known_args()


device = torch.device(args.device)
# experiment_description = args.experiment_description
sourcedata = args.pretrain_dataset
targetdata = args.target_dataset
experiment_description = str(sourcedata)+'_2_'+str(targetdata)


method = 'Time-Freq Consistency' # 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description

logs_save_dir = args.logs_save_dir
os.makedirs(logs_save_dir, exist_ok=True)


exec(f'from config_files.{sourcedata}_Configs import Config as Configs')
exec(f'from config_files.{targetdata}_Configs import Config as Configs_target')
configs = Configs() # THis is OK???
configs_target = Configs_target() # THis is OK???

if args.batch is not None:
        configs.target_batch_size = args.batch
        configs_target.target_batch_size = args.batch

# # ##### fix random seeds for reproducibility ########
SEED = args.seed
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description, training_mode + f"_seed_{SEED}")
# 'experiments_logs/Exp1/run1/train_linear_seed_0'
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0


# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
# 'experiments_logs/Exp1/run1/train_linear_seed_0/logs_14_04_2022_15_13_12.log'
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Pre-training Dataset: {sourcedata}')
logger.debug(f'Target (fine-tuning) Dataset: {targetdata}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
sourcedata_path = f"../datasets/{sourcedata}"  # './data/Epilepsy'
targetdata_path = f"../datasets/{targetdata}"
# for self-supervised, the data are augmented here. Only self-supervised learning need augmentation
subset = False # if subset= true, use a subset for debugging.
train_dl, valid_dl, test_dl = data_generator(sourcedata_path, targetdata_path, configs, configs_target, training_mode, subset = subset, percent=args.percent)
logger.debug("Data loaded ...")
logger.debug(f"Training mode = {training_mode}")

n_pipes = 2 if args.single_pipe is None else 1
TFC_model = TFC(configs).to(device)
classifier = target_classifier(configs_target, n_pipes).to(device)

temporal_contr_model = None #TC(configs, device).to(device)


if training_mode == "fine_tune":
    # load saved model of this experiment
    load_from = os.path.join(os.path.join(logs_save_dir, experiment_description, run_description,
    f"pre_train_seed_{SEED}", "saved_models")) # 'experiments_logs/Exp1/run1/self_supervised_seed_0/saved_models'
    pretrained_weights = os.path.join(load_from, "ckp_last.pt")
    logger.debug(f"Loading model from {pretrained_weights}!")
    chkpoint = torch.load(pretrained_weights, map_location=device) # two saved models: ['model_state_dict', 'temporal_contr_model_state_dict']


    pretrained_dict = chkpoint["model_state_dict"] # Time domain parameters
    model_dict = TFC_model.state_dict()
    # pretrained_dict = remove_logits(pretrained_dict)
    model_dict.update(pretrained_dict)
    TFC_model.load_state_dict(model_dict)


model_optimizer = torch.optim.Adam(TFC_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)
classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=3e-4, betas=(configs_target.beta1, configs_target.beta2), weight_decay=3e-4)
temporal_contr_optimizer = None # torch.optim.Adam(temporal_contr_model.parameters(), lr=configs.lr, betas=(configs.beta1, configs.beta2), weight_decay=3e-4)

#if training_mode == "pre_train":  # to do it only once
#    copy_Files(os.path.join(logs_save_dir, experiment_description, run_description), sourcedata)

# Trainer
Trainer(TFC_model,  temporal_contr_model, model_optimizer, temporal_contr_optimizer, train_dl, valid_dl, test_dl, device,
        logger, configs, configs_target, experiment_log_dir, training_mode, model_F=None, model_F_optimizer = None,
        classifier=classifier, classifier_optimizer=classifier_optimizer, epochs=args.epochs, pipes=args.single_pipe)

logger.debug(f"Training time is : {datetime.now()-start_time}")
