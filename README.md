# RouteNet Reliability Gate

Reliability-gated, distribution-aware queue aggregation for RouteNet-Fermi scalability experiments.

The model preserves the original sum aggregation channel and augments it with queue-message distribution statistics. A deterministic gate suppresses the auxiliary encoding when too few flow messages reach a queue:

```text
n_eff = max(N_q - 1, 0)
g_q   = n_eff / (n_eff + tau)
M_q   = concat(sum_i x_qi, g_q * e_q)
```

The default configuration uses `tau = 4` and includes seed `1234` among the reproducible training seeds.

## Layout

- `Scalability/weighted_queue_model.py`: RouteNet-Fermi queue-message backbone.
- `Scalability/train_scalability.py`: shared scalability training pipeline.
- `Scalability/ReliabilityGatedDistributionAwareQueue/`: reliability-gated model, launchers, configuration, tests, and result summarization.

## Requirements

- Python 3.6+
- TensorFlow 2.6
- NetworkX 2.5.1
- NumPy
- Matplotlib

The official RouteNet-Fermi source tree and scalability dataset are external dependencies and are not included. Place `RouteNet-Fermi` beside this repository, or pass its path explicitly with `--source-root`.

## Quick check

```bash
cd Scalability/ReliabilityGatedDistributionAwareQueue
python smoke_test.py
python run_five_seeds.py --source-root /path/to/RouteNet-Fermi --dry-run
```

See the model-specific README in that directory for the full training workflow.

