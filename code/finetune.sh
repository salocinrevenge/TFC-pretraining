# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar_original --target_dataset KuHar_original
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar_original_6 --target_dataset KuHar_original_6
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar --target_dataset KuHar
# CUDA_VISIBLE_DEVICES=4 python main.py --training_mode fine_tune --pretrain_dataset UCI_original --target_dataset UCI_original
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset SleepEEG --target_dataset Epilepsy
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset SleepEEG --target_dataset UCI_original
CUDA_VISIBLE_DEVICES=4 python main.py --training_mode fine_tune --pretrain_dataset UCI_original --target_dataset UCI_original --percent 1 --epochs 40 --batch 42