# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode pre_train --pretrain_dataset KuHar_original --target_dataset KuHar_original
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode pre_train --pretrain_dataset KuHar_original_norm --target_dataset KuHar_original_norm
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode pre_train --pretrain_dataset KuHar --target_dataset KuHar
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset Epilepsy
# CUDA_VISIBLE_DEVICES=4 python main.py --training_mode pre_train --pretrain_dataset SleepEEG --target_dataset UCI_original
CUDA_VISIBLE_DEVICES=4 python main.py --training_mode pre_train --pretrain_dataset UCI_original --target_dataset UCI_original --epochs 40 --seed 3
