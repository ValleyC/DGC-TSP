"""
Training script for DGC-TSP.

Usage:
    python train.py --data_path data/tsp100_train.txt --num_clusters 10 --epochs 100
"""

import argparse
import os
import time
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from dgc_tsp import DGCTSP
from dgc_tsp.utils import TSPDataset, collate_tsp_batch, compute_tour_length


def parse_args():
    parser = argparse.ArgumentParser(description='Train DGC-TSP model')

    # Data
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to training data file')
    parser.add_argument('--val_data_path', type=str, default=None,
                        help='Path to validation data file')

    # Model
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Hidden dimension')
    parser.add_argument('--num_clusters', type=int, default=10,
                        help='Number of clusters')
    parser.add_argument('--num_egnn_layers', type=int, default=4,
                        help='Number of EGNN layers')
    parser.add_argument('--num_diffusion_layers', type=int, default=3,
                        help='Number of diffusion layers')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='Number of attention heads')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Number of diffusion inference steps')

    # Loss weights
    parser.add_argument('--lambda_clustering', type=float, default=1.0,
                        help='Weight for clustering loss')
    parser.add_argument('--lambda_diffusion', type=float, default=1.0,
                        help='Weight for diffusion loss')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Temperature for cluster assignment')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['none', 'cosine', 'step'],
                        help='Learning rate scheduler')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Warmup epochs')

    # Logging
    parser.add_argument('--log_interval', type=int, default=100,
                        help='Log every N batches')
    parser.add_argument('--val_interval', type=int, default=1,
                        help='Validate every N epochs')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name')

    # Other
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    return parser.parse_args()


def set_seed(seed):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_clustering_loss = 0.0
    total_diffusion_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        # Move to device
        coords = batch['coords'].to(device)
        adj = batch['adj'].to(device)
        tour = batch['tour'].to(device)

        # Forward pass
        optimizer.zero_grad()
        outputs = model(coords, adj=adj, tour=tour)

        # Backward pass
        loss = outputs['losses']['total']
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Accumulate losses
        total_loss += loss.item()
        total_clustering_loss += outputs['losses']['clustering']['total'].item()
        total_diffusion_loss += outputs['losses']['diffusion']['total'].item()
        num_batches += 1

        # Logging
        if (batch_idx + 1) % args.log_interval == 0:
            avg_loss = total_loss / num_batches
            avg_cluster = total_clustering_loss / num_batches
            avg_diffusion = total_diffusion_loss / num_batches
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(dataloader)}] "
                  f"Loss: {avg_loss:.4f} (Cluster: {avg_cluster:.4f}, Diff: {avg_diffusion:.4f})")

    return {
        'loss': total_loss / num_batches,
        'clustering_loss': total_clustering_loss / num_batches,
        'diffusion_loss': total_diffusion_loss / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, device, args):
    """Validate model."""
    model.eval()
    total_loss = 0.0
    total_gap = 0.0
    num_samples = 0

    for batch in dataloader:
        coords = batch['coords'].to(device)
        adj = batch['adj'].to(device)
        tour = batch['tour'].to(device)

        # Compute loss
        outputs = model(coords, adj=adj, tour=tour)
        total_loss += outputs['losses']['total'].item()

        # Evaluate solution quality (first sample in batch)
        batch_size = coords.shape[0]
        for b in range(batch_size):
            try:
                solution = model.solve(coords[b:b+1])
                pred_tour = solution['tour'].cpu().numpy()
                gt_tour = tour[b].cpu().numpy()

                # Compute lengths
                coords_np = coords[b].cpu().numpy()
                pred_length = compute_tour_length(coords_np, pred_tour)
                gt_length = compute_tour_length(coords_np, gt_tour)

                gap = (pred_length - gt_length) / gt_length * 100
                total_gap += gap
                num_samples += 1
            except Exception as e:
                print(f"Warning: Failed to solve instance: {e}")
                continue

    avg_loss = total_loss / len(dataloader)
    avg_gap = total_gap / max(num_samples, 1)

    return {
        'loss': avg_loss,
        'gap': avg_gap,
        'num_samples': num_samples,
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup experiment name
    if args.exp_name is None:
        args.exp_name = f"dgc_tsp_k{args.num_clusters}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Create save directory
    save_dir = os.path.join(args.save_dir, args.exp_name)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving to: {save_dir}")

    # Load data
    print(f"Loading training data from: {args.data_path}")
    train_dataset = TSPDataset(args.data_path)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_tsp_batch,
    )
    print(f"Training samples: {len(train_dataset)}")

    val_loader = None
    if args.val_data_path:
        print(f"Loading validation data from: {args.val_data_path}")
        val_dataset = TSPDataset(args.val_data_path)
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,  # Single instance for evaluation
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_tsp_batch,
        )
        print(f"Validation samples: {len(val_dataset)}")

    # Create model
    model = DGCTSP(
        hidden_dim=args.hidden_dim,
        num_clusters=args.num_clusters,
        num_egnn_layers=args.num_egnn_layers,
        num_diffusion_layers=args.num_diffusion_layers,
        num_heads=args.num_heads,
        num_inference_steps=args.num_inference_steps,
        lambda_clustering=args.lambda_clustering,
        lambda_diffusion=args.lambda_diffusion,
        temperature=args.temperature,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Learning rate scheduler
    if args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs,
        )
    elif args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=30,
            gamma=0.5,
        )
    else:
        scheduler = None

    # Training loop
    best_gap = float('inf')
    best_epoch = 0

    print("\nStarting training...")
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Warmup learning rate
        if epoch <= args.warmup_epochs:
            warmup_lr = args.lr * epoch / args.warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = warmup_lr

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, args)

        # Update scheduler
        if scheduler is not None and epoch > args.warmup_epochs:
            scheduler.step()

        epoch_time = time.time() - start_time

        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train Loss: {train_metrics['loss']:.4f}")
        print(f"  Clustering: {train_metrics['clustering_loss']:.4f}")
        print(f"  Diffusion: {train_metrics['diffusion_loss']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Validation
        if val_loader is not None and epoch % args.val_interval == 0:
            val_metrics = validate(model, val_loader, device, args)
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val Gap: {val_metrics['gap']:.2f}%")

            # Save best model
            if val_metrics['gap'] < best_gap:
                best_gap = val_metrics['gap']
                best_epoch = epoch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_gap': val_metrics['gap'],
                    'args': args,
                }, os.path.join(save_dir, 'best_model.pt'))
                print(f"  New best model saved! Gap: {best_gap:.2f}%")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': args,
            }, os.path.join(save_dir, f'checkpoint_epoch{epoch}.pt'))

    print(f"\nTraining complete!")
    print(f"Best validation gap: {best_gap:.2f}% at epoch {best_epoch}")

    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'args': args,
    }, os.path.join(save_dir, 'final_model.pt'))


if __name__ == '__main__':
    main()
