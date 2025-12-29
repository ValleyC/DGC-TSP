"""
GEPNet Training Script for TSP.

TSP-EGNN-ICAM: E(2)-Equivariant partition + ICAM sub-solver.

Key innovations over UDC:
1. EGNN backbone: No coordinate_transformation needed
2. Invariant features by construction
3. State-aware partition with first, last, visited embeddings (no depot for TSP)
"""

import os
import sys

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # For TSProblemDef
sys.path.insert(0, "../..")  # For utils

import logging
from utils.utils import create_logger, copy_all_src
from TSPTrainerPartition import TSPTrainerPartition as Trainer

# Environment parameters
env_params = {
    'problem_size_low': 500,      # Minimum problem size
    'problem_size_high': 1000,    # Maximum problem size
    'problem_size': 100,          # Sub-problem size
    'pomo_size': 50,              # POMO augmentation size
    'sample_size': 40,            # Number of partition samples
    'fs_sample_size': 10,         # First-stage sample size (curriculum)
}

# GEPNet partition model parameters
model_p_params = {
    'embedding_dim': 64,  # Same as UDC
    'depth': 12,          # Same as UDC
    'use_egnn': True,     # Our contribution: use EGNN instead of EmbNet
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
}

logger_params = {
    'log_file': {
        'desc': 'gepnet_tsp_n500_n1000',
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
    logger.info('GEPNet: E(2)-Equivariant Partition Network for TSP')
    logger.info('=' * 60)
    logger.info(f'DEBUG_MODE: {DEBUG_MODE}')
    logger.info(f'USE_CUDA: {USE_CUDA}, CUDA_DEVICE_NUM: {CUDA_DEVICE_NUM}')
    logger.info(f'Problem size: {env_params["problem_size_low"]} - {env_params["problem_size_high"]}')
    logger.info(f'Sub-problem size: {env_params["problem_size"]}')
    logger.info(f'EGNN depth: {model_p_params["depth"]}, embedding dim: {model_p_params["embedding_dim"]}')
    logger.info(f'Use EGNN: {model_p_params["use_egnn"]}')
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
