"""
Data generation script for DGC-TSP.

Generates TSP instances with optimal or near-optimal tours.

Usage:
    python scripts/generate_data.py --num_samples 100000 --num_nodes 100 --output data/tsp100_train.txt
"""

import argparse
import os
import sys
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dgc_tsp.utils import (
    generate_tsp_instance,
    nearest_neighbor_tour,
    two_opt,
    compute_tour_length,
)


def parse_args():
    parser = argparse.ArgumentParser(description='Generate TSP training data')

    parser.add_argument('--num_samples', type=int, default=100000,
                        help='Number of samples to generate')
    parser.add_argument('--num_nodes', type=int, default=100,
                        help='Number of nodes per instance')
    parser.add_argument('--output', type=str, required=True,
                        help='Output file path')
    parser.add_argument('--solver', type=str, default='two_opt',
                        choices=['nearest_neighbor', 'two_opt', 'lkh', 'concorde'],
                        help='Solver to use for generating tours')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=None,
                        help='Number of parallel workers')
    parser.add_argument('--two_opt_iterations', type=int, default=1000,
                        help='Max 2-opt iterations')

    return parser.parse_args()


def solve_instance(args_tuple):
    """Solve a single TSP instance."""
    idx, num_nodes, solver, seed, two_opt_iterations = args_tuple

    # Generate instance
    np.random.seed(seed + idx)
    coords = generate_tsp_instance(num_nodes)

    # Solve
    if solver == 'nearest_neighbor':
        tour = nearest_neighbor_tour(coords)
    elif solver == 'two_opt':
        tour = nearest_neighbor_tour(coords)
        tour = two_opt(coords, tour, max_iterations=two_opt_iterations)
    elif solver == 'lkh':
        try:
            from dgc_tsp.utils import solve_tsp_lkh
            tour, _ = solve_tsp_lkh(coords)
        except ImportError:
            tour = nearest_neighbor_tour(coords)
            tour = two_opt(coords, tour)
    elif solver == 'concorde':
        try:
            from dgc_tsp.utils import solve_tsp_concorde
            tour, _ = solve_tsp_concorde(coords)
        except ImportError:
            tour = nearest_neighbor_tour(coords)
            tour = two_opt(coords, tour)
    else:
        raise ValueError(f"Unknown solver: {solver}")

    # Compute length
    length = compute_tour_length(coords, tour)

    return coords, tour, length


def format_instance(coords, tour):
    """Format instance as a single line."""
    n = len(coords)

    # Format: x1 y1 x2 y2 ... xn yn tour_idx1 tour_idx2 ... tour_idxn
    parts = []

    # Coordinates
    for i in range(n):
        parts.append(f"{coords[i, 0]:.6f}")
        parts.append(f"{coords[i, 1]:.6f}")

    # Tour
    for i in range(n):
        parts.append(str(tour[i]))

    return ' '.join(parts)


def main():
    args = parse_args()
    np.random.seed(args.seed)

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)

    # Number of workers
    num_workers = args.num_workers or max(1, cpu_count() - 1)
    print(f"Generating {args.num_samples} TSP{args.num_nodes} instances")
    print(f"Using solver: {args.solver}")
    print(f"Workers: {num_workers}")

    # Prepare arguments
    instance_args = [
        (i, args.num_nodes, args.solver, args.seed, args.two_opt_iterations)
        for i in range(args.num_samples)
    ]

    # Generate instances
    lengths = []
    with open(args.output, 'w') as f:
        if num_workers > 1:
            with Pool(num_workers) as pool:
                for coords, tour, length in tqdm(
                    pool.imap(solve_instance, instance_args),
                    total=args.num_samples,
                    desc="Generating"
                ):
                    line = format_instance(coords, tour)
                    f.write(line + '\n')
                    lengths.append(length)
        else:
            for instance_arg in tqdm(instance_args, desc="Generating"):
                coords, tour, length = solve_instance(instance_arg)
                line = format_instance(coords, tour)
                f.write(line + '\n')
                lengths.append(length)

    # Statistics
    lengths = np.array(lengths)
    print(f"\nGenerated {args.num_samples} instances")
    print(f"Output: {args.output}")
    print(f"Tour length statistics:")
    print(f"  Mean: {lengths.mean():.4f}")
    print(f"  Std:  {lengths.std():.4f}")
    print(f"  Min:  {lengths.min():.4f}")
    print(f"  Max:  {lengths.max():.4f}")


if __name__ == '__main__':
    main()
