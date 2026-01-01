"""
EDGC Training Script for CVRP.

Equivariant Deep Graph Clustering for CVRP partition.

Key innovations:
1. One-shot O(N) clustering instead of O(N^2) autoregressive
2. InfoNCE + KL + Capacity + REINFORCE hybrid loss
3. EGNN backbone for E(2)-equivariance
4. Learnable cluster centers with Student-t assignment

References:
- RGC: InfoNCE contrastive loss + RL
- SDCN: Learnable cluster centers + Student-t
- SCGC: Dual projection with noise augmentation
"""

import os
import sys

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")
sys.path.insert(0, "../..")

import logging
from utils.utils import create_logger, copy_all_src
from EDGCTrainer import EDGCTrainer as Trainer

# Environment parameters
env_params = {
    'problem_size_low': 500,
    'problem_size_high': 1000,
    'sub_size': 100,
    'pomo_size': 50,
    'sample_size': 40,
}

# EDGC partition model parameters
model_p_params = {
    'embedding_dim': 64,
    'depth': 12,
    'projection_dim': 128,      # Dimension for dual projection heads
    'max_clusters': 50,         # Maximum number of clusters
}

# Sub-solver model parameters (ICAM)
model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** 0.5,
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 50,
    'ff_hidden_dim': 512,
    'eval_type': 'softmax',
}

# Optimizer parameters
optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 0
    },
    'optimizer_p': {
        'lr': 1e-4,
        'weight_decay': 0
    },
    'scheduler': {
        'milestones': [3001],
        'gamma': 0.1
    }
}

# Trainer parameters
trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 500,
    'train_episodes': 1000,
    'train_batch_size': 1,
    'logging': {
        'model_save_interval': 10,
    },
    'model_load': {
        't_enable': False,
        't_path': './',
        't_epoch': 0,
        'p_enable': False,
        'p_path': './',
        'p_epoch': 0,
    },
    # EDGC loss weights
    'w_infonce': 1.0,       # InfoNCE contrastive loss
    'w_kl': 0.1,            # KL clustering loss
    'w_capacity': 10.0,     # Capacity constraint penalty
    'w_balance': 0.01,      # Cluster balance loss
    'w_reinforce': 1.0,     # REINFORCE from solver reward
    'temperature': 0.5,     # InfoNCE temperature
}

logger_params = {
    'log_file': {
        'desc': 'edgc_cvrp_n500_n1000',
        'filename': 'log.txt'
    }
}


def main():
    if DEBUG_MODE:
        trainer_params['epochs'] = 2
        trainer_params['train_episodes'] = 10
        trainer_params['train_batch_size'] = 1

    create_logger(**logger_params)

    logger = logging.getLogger('root')
    logger.info('=' * 60)
    logger.info('EDGC: Equivariant Deep Graph Clustering for CVRP')
    logger.info('=' * 60)
    logger.info(f'DEBUG_MODE: {DEBUG_MODE}')
    logger.info(f'USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}')
    logger.info(f'Problem size: {env_params["problem_size_low"]} - {env_params["problem_size_high"]}')
    logger.info(f'Sub-problem size: {env_params["sub_size"]}')
    logger.info(f'EGNN depth: {model_p_params["depth"]}, embedding dim: {model_p_params["embedding_dim"]}')
    logger.info(f'Projection dim: {model_p_params["projection_dim"]}, max clusters: {model_p_params["max_clusters"]}')
    logger.info('Loss weights:')
    logger.info(f'  InfoNCE: {trainer_params["w_infonce"]}, KL: {trainer_params["w_kl"]}')
    logger.info(f'  Capacity: {trainer_params["w_capacity"]}, Balance: {trainer_params["w_balance"]}')
    logger.info(f'  REINFORCE: {trainer_params["w_reinforce"]}')
    logger.info('=' * 60)

    trainer = Trainer(
        env_params=env_params,
        model_params=model_params,
        model_p_params=model_p_params,
        optimizer_params=optimizer_params,
        trainer_params=trainer_params
    )

    copy_all_src(trainer.result_folder)
    trainer.run()


if __name__ == "__main__":
    main()
