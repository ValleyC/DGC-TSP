# GEPNet: Geometric Equivariant Partition Network

E(2)-Equivariant Divide-and-Conquer for Large-Scale Combinatorial Optimization.

## Key Innovation

GEPNet replaces UDC's AGNN backbone with **E(2)-equivariant EGNN**, achieving:
1. **No coordinate normalization** - UDC's `coordinate_transformation()` is unnecessary
2. **No 8-fold augmentation** - Partition model is inherently rotation/translation invariant
3. **State-aware partition** - Considers depot, first, last, and visited node embeddings

## Repository Structure (Following UDC Pattern)

```
GEPNet/
├── CVRProblemDef.py           # CVRP problem generation
├── utils/
│   └── utils.py               # Training utilities
│
├── CVRP-EGNN-ICAM/            # CVRP with EGNN partition + ICAM sub-solver
│   ├── PartitionModel.py      # GEPNet partition (EGNN + state-aware head)
│   ├── CVRPEnv.py             # CVRP environment
│   ├── CVRPModel.py           # ICAM sub-solver
│   ├── CVRPTrainerPartition.py # Training logic
│   └── train_gepnet.py        # Training script
│
├── TSP-EGNN-ICAM/             # TSP with EGNN partition + ICAM sub-solver
│   ├── PartitionModel.py      # GEPNet partition (EGNN + state-aware head, no depot)
│   ├── TSPEnv.py              # TSP environment
│   ├── TSPModel.py            # ICAM sub-solver (AFT-based)
│   ├── TSPTrainerPartition.py # Training logic
│   └── train_gepnet.py        # Training script
│
├── PCTSP-EGNN-POMO/           # (Coming soon)
└── ...
```

## Theoretical Contribution

### Why EGNN for Partition?

UDC's partition network is **not equivariant** - it uses:
```python
# UDC's hacky normalization
def coordinate_transformation(self, x):
    input[:, :, 0] -= min_x  # shift
    input[:, :, 1] -= min_y
    input /= scale_degree   # scale
```

GEPNet uses **EGNN** which is equivariant by construction:
- Message passing uses distances (invariant)
- Coordinate updates are equivariant
- No normalization needed!

### State-Aware Partition Query

For CVRP, the partition query combines:
```python
q = x_depot + x_first + x_last + mean(x_visited)
```
This captures:
- **Depot**: Where vehicles return
- **First/Last**: Boundaries of current sub-path
- **Visited**: Already assigned nodes

## Usage

### Training CVRP
```bash
cd CVRP-EGNN-ICAM
python train_gepnet.py
```

### Configuration
Edit `train_gepnet.py` to modify:
- `problem_size_low/high`: Problem size range (default: 500-1000)
- `sub_size`: Sub-problem size (default: 100)
- `model_p_params['depth']`: EGNN depth (default: 12)
- `model_p_params['use_egnn']`: True for EGNN, False for UDC's EmbNet

## Comparison: GEPNet vs UDC

| Aspect | UDC (AGNN) | GEPNet (EGNN) |
|--------|------------|---------------|
| Backbone | Standard GNN | E(2)-Equivariant GNN |
| Invariance | `coordinate_transformation` | By construction |
| Augmentation | 8-fold | Not needed |
| Node features | Raw coordinates | Polar (r, θ) + demand |
| State query | first + last + visited | depot + first + last + visited |

## Citation

Based on:
- **UDC** (NeurIPS 2024): Unified Divide-and-Conquer
- **EGNN**: E(n)-Equivariant Graph Neural Networks
- **GLOP** (AAAI 2024): Global-Local Policy for VRP
