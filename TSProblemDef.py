"""
TSP Problem Definition.

Generates random TSP instances.
"""

import torch
import numpy as np


def get_random_problems(batch_size, problem_size):
    """Generate random TSP instances with uniform coordinates in [0, 1]^2."""
    problems = torch.rand(size=(batch_size, problem_size, 2))
    return problems


def augment_xy_data_by_8_fold(problems):
    """
    8-fold data augmentation via rotation and reflection.

    Note: NOT needed for GEPNet (equivariant by design),
    but kept for ablation studies and comparison with UDC.
    """
    x = problems[:, :, [0]]
    y = problems[:, :, [1]]

    dat1 = torch.cat((x, y), dim=2)
    dat2 = torch.cat((1 - x, y), dim=2)
    dat3 = torch.cat((x, 1 - y), dim=2)
    dat4 = torch.cat((1 - x, 1 - y), dim=2)
    dat5 = torch.cat((y, x), dim=2)
    dat6 = torch.cat((1 - y, x), dim=2)
    dat7 = torch.cat((y, 1 - x), dim=2)
    dat8 = torch.cat((1 - y, 1 - x), dim=2)

    aug_problems = torch.cat((dat1, dat2, dat3, dat4, dat5, dat6, dat7, dat8), dim=0)
    return aug_problems
