"""
Inference script for DGC-TSP.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pt --data_path data/tsp100_test.txt
"""

import argparse
import os
import time

import torch
import numpy as np
from tqdm import tqdm

from dgc_tsp import DGCTSP
from dgc_tsp.utils import TSPDataset, compute_tour_length, two_opt, collate_tsp_batch


def parse_args():
    parser = argparse.ArgumentParser(description='DGC-TSP Inference')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to test data file')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for predictions')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='Number of samples to evaluate (default: all)')
    parser.add_argument('--use_2opt', action='store_true',
                        help='Apply 2-opt refinement')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser.parse_args()


def main():
    args = parse_args()

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_args = checkpoint['args']

    # Create model
    model = DGCTSP(
        hidden_dim=model_args.hidden_dim,
        num_clusters=model_args.num_clusters,
        num_egnn_layers=model_args.num_egnn_layers,
        num_diffusion_layers=model_args.num_diffusion_layers,
        num_heads=model_args.num_heads,
        num_inference_steps=model_args.num_inference_steps,
        lambda_clustering=model_args.lambda_clustering,
        lambda_diffusion=model_args.lambda_diffusion,
        temperature=model_args.temperature,
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model from epoch {checkpoint['epoch']}")

    # Load data
    print(f"Loading data: {args.data_path}")
    dataset = TSPDataset(args.data_path)

    if args.num_samples:
        num_samples = min(args.num_samples, len(dataset))
    else:
        num_samples = len(dataset)
    print(f"Evaluating {num_samples} samples")

    # Inference
    gaps = []
    pred_lengths = []
    gt_lengths = []
    times = []

    results = []

    with torch.no_grad():
        for idx in tqdm(range(num_samples), desc="Evaluating"):
            item = dataset[idx]
            coords = item['coords'].unsqueeze(0).to(device)
            gt_tour = item['tour'].numpy()

            # Solve
            start_time = time.time()
            solution = model.solve(coords, return_clusters=True)
            inference_time = time.time() - start_time

            pred_tour = solution['tour'].cpu().numpy()

            # Optional 2-opt refinement
            coords_np = item['coords'].numpy()
            if args.use_2opt:
                pred_tour = two_opt(coords_np, pred_tour)

            # Compute lengths
            pred_length = compute_tour_length(coords_np, pred_tour)
            gt_length = compute_tour_length(coords_np, gt_tour)

            gap = (pred_length - gt_length) / gt_length * 100

            gaps.append(gap)
            pred_lengths.append(pred_length)
            gt_lengths.append(gt_length)
            times.append(inference_time)

            results.append({
                'idx': idx,
                'pred_tour': pred_tour,
                'pred_length': pred_length,
                'gt_length': gt_length,
                'gap': gap,
                'time': inference_time,
                'cluster_assignments': solution['cluster_assignments'].cpu().numpy()[0],
            })

    # Statistics
    gaps = np.array(gaps)
    pred_lengths = np.array(pred_lengths)
    gt_lengths = np.array(gt_lengths)
    times = np.array(times)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Number of samples: {num_samples}")
    print(f"\nTour Length:")
    print(f"  GT Mean:   {gt_lengths.mean():.4f}")
    print(f"  Pred Mean: {pred_lengths.mean():.4f}")
    print(f"\nOptimality Gap:")
    print(f"  Mean: {gaps.mean():.2f}%")
    print(f"  Std:  {gaps.std():.2f}%")
    print(f"  Min:  {gaps.min():.2f}%")
    print(f"  Max:  {gaps.max():.2f}%")
    print(f"\nInference Time:")
    print(f"  Mean: {times.mean() * 1000:.1f} ms")
    print(f"  Total: {times.sum():.1f} s")

    # Save results
    if args.output:
        print(f"\nSaving results to: {args.output}")
        with open(args.output, 'w') as f:
            f.write("idx,pred_length,gt_length,gap,time\n")
            for r in results:
                f.write(f"{r['idx']},{r['pred_length']:.6f},{r['gt_length']:.6f},"
                        f"{r['gap']:.4f},{r['time']:.6f}\n")


if __name__ == '__main__':
    main()
