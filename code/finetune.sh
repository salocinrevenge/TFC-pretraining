# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar_original --target_dataset KuHar_original
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar_original_6 --target_dataset KuHar_original_6
# CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset KuHar --target_dataset KuHar
CUDA_VISIBLE_DEVICES=5 python main.py --training_mode fine_tune --pretrain_dataset UCI_original --target_dataset UCI_original --device cuda