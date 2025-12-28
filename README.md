# DGC-TSP: Deep Graph Clustering for Hierarchical TSP

This repository implements a novel approach combining Deep Graph Clustering (DGC) with hierarchical diffusion for solving large-scale Traveling Salesman Problems.

## Motivation

Traditional divide-and-conquer approaches for large-scale TSP (like GLOP) use fixed partitioning strategies that may not align with the optimal tour structure. Our approach learns to partition the graph in a way that:

1. **Minimizes inter-cluster edges** in the optimal tour
2. **Creates balanced clusters** for efficient parallel processing
3. **Adapts to problem structure** rather than using geometric proximity alone

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     DGC-TSP Pipeline                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. EGNN Encoder                                            │
│     ├── Input: Node coordinates (n × 2)                     │
│     ├── Process: E(n)-equivariant message passing           │
│     └── Output: Node embeddings h ∈ R^(n × d)               │
│                                                             │
│  2. Learned Clustering Module                               │
│     ├── Input: Node embeddings h                            │
│     ├── Process: Soft cluster assignment via attention      │
│     └── Output: Cluster assignments C ∈ R^(n × k)           │
│                                                             │
│  3. Hierarchical Diffusion                                  │
│     ├── Level 1 (Coarse): Inter-cluster routing             │
│     ├── Level 2 (Fine): Intra-cluster routing               │
│     └── Cross-level message passing                         │
│                                                             │
│  4. Tour Assembly                                           │
│     ├── Merge cluster solutions                             │
│     └── Optional: 2-opt refinement                          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## Key Components

### 1. Deep Graph Clustering (DGC)

Unlike RGC which uses reinforcement learning to determine cluster count, our DGC:
- Uses self-supervised contrastive learning
- Learns cluster assignments that align with TSP tour structure
- Provides differentiable soft assignments for end-to-end training

### 2. Tour-Aware Clustering Loss

```
L_cluster = L_contrastive + λ₁ * L_tour_alignment + λ₂ * L_balance
```

Where:
- `L_contrastive`: InfoNCE loss for learning good representations
- `L_tour_alignment`: Encourages clusters to minimize edge cuts in optimal tour
- `L_balance`: Ensures balanced cluster sizes

### 3. Hierarchical Diffusion

Two-level diffusion process:
- **Coarse level**: Learns inter-cluster connections
- **Fine level**: Learns intra-cluster routing conditioned on coarse structure

## Installation

```bash
pip install torch torch-geometric numpy matplotlib
```

## Usage

### Training

```bash
# Generate training data
python scripts/generate_data.py --num_samples 100000 --num_nodes 100 --output data/tsp100_train.txt

# Train DGC-TSP model
python train.py --data_path data/tsp100_train.txt --num_clusters 10 --epochs 100
```

### Inference

```bash
python inference.py --checkpoint checkpoints/best_model.pt --data_path data/tsp100_test.txt
```

## Comparison with GLOP

| Aspect | GLOP | DGC-TSP |
|--------|------|---------|
| Partitioning | Fixed sliding window | Learned clustering |
| Coordination | Sequential iteration | Hierarchical diffusion |
| Equivariance | Data augmentation | Built-in E(n) |
| Parallelization | Limited | High (clusters independent) |

## References

- GLOP: Learning Global-Local Policies for Large-Scale TSP
- EDISCO: Equivariant Diffusion for Combinatorial Optimization
- RGC: Reinforcement Graph Clustering
- Deep Graph Clustering (Liu et al., 2022, 2023)

## License

MIT License
