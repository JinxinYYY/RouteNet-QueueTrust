# RouteNet QueueTrust

QueueTrust is a reliability-aware queue-message aggregation method for RouteNet-Fermi. It preserves the original sum aggregation and adds distribution information only when the messages observed at a queue provide enough statistical support.

## Motivation

Distribution statistics such as variance and maximum can help distinguish queues with similar aggregate traffic but different flow compositions. These statistics are unstable when a queue contains very few incoming flow messages. QueueTrust controls that uncertainty with a deterministic reliability factor.

For a queue with `N_q` incoming messages:

```text
n_eff = max(N_q - 1, 0)
g_q   = n_eff / (n_eff + tau)
M_q   = concat(sum_i x_qi, g_q * e_q)
```

Here, `e_q` is the learned distribution encoding and `tau` controls how quickly the auxiliary channel becomes trusted. The default is `tau = 4`. The original sum channel is never attenuated.

## Results

QueueTrust achieves the lowest average MAPE in the scalability test. Because MAPE is an error metric, lower values are better.

| Model | Average test MAPE | Absolute reduction vs. RouteNet-Fermi | Relative error reduction vs. RouteNet-Fermi |
|---|---:|---:|---:|
| RouteNet-Fermi | 1.060% | — | — |
| Distribution-Aware Queue | 0.942% | 0.118 percentage points | 11.13% |
| **QueueTrust** | **0.823%** | **0.237 percentage points** | **22.36%** |

QueueTrust therefore reduces average prediction error by **22.36%** relative to the original RouteNet-Fermi model. It also improves on the ungated Distribution-Aware Queue model by **0.119 percentage points**, corresponding to a **12.63%** relative error reduction.

The advantage becomes more pronounced on larger topologies. At approximately 300 nodes, QueueTrust records about **1.20% MAPE**, compared with about **2.13%** for RouteNet-Fermi—an error reduction of roughly **44%**. These 300-node values are read from the plotted topology-size curve and are therefore approximate.

## Repository structure

```text
Scalability/
├── train_scalability.py
├── weighted_queue_model.py
└── ReliabilityGatedDistributionAwareQueue/
    ├── experiment_config.json
    ├── reliability_gated_model.py
    ├── train_reliability_gated_scalability.py
    ├── run_five_seeds.py
    ├── summarize_five_seeds.py
    ├── smoke_test.py
    ├── start_five_seeds.sh
    ├── status_five_seeds.sh
    └── run_task.ps1
```

Datasets, checkpoints, logs, figures, and generated experiment outputs are intentionally excluded.

## Requirements

- Python 3.6+
- TensorFlow 2.6
- NetworkX 2.5.1
- NumPy
- Matplotlib

Install the additional pinned dependency with:

```bash
pip install -r requirements.txt
```

TensorFlow is not pinned in `requirements.txt` because the original experiments used the TensorFlow 2.6 GPU container image.

## Dataset and upstream code

QueueTrust uses the official RouteNet-Fermi scalability dataset and its data generator. The upstream `RouteNet-Fermi` directory is expected beside this repository:

```text
workspace/
├── RouteNet-Fermi/
└── RouteNet-QueueTrust/
```

You may place it elsewhere and provide the location through `--source-root`.

## Quick verification

```bash
cd Scalability/ReliabilityGatedDistributionAwareQueue
python smoke_test.py
```

Preview the seed `1234` command without starting training:

```bash
python run_five_seeds.py \
  --source-root /path/to/RouteNet-Fermi \
  --seeds 1234 \
  --dry-run
```

## Train with seed 1234

```bash
python run_five_seeds.py \
  --source-root /path/to/RouteNet-Fermi \
  --seeds 1234
```

The full reproducibility configuration is stored in `experiment_config.json`. The launcher also supports the complete seed set `1234, 2026, 3407, 5678, 9012`.

On Linux, the background launcher can be used after setting `SOURCE_ROOT`:

```bash
export SOURCE_ROOT=/path/to/RouteNet-Fermi
bash start_five_seeds.sh --seeds 1234
bash status_five_seeds.sh
```

On Windows:

```powershell
.\run_task.ps1 smoke
.\run_task.ps1 dryrun
```

## Outputs

Training artifacts are written to:

```text
runs/seed_<SEED>/attempt_<TIMESTAMP>/
```

The launcher preserves incomplete attempts and skips a seed only after validating its result, state, configuration, and per-topology evaluation files.

## License

This project is released under the Apache License 2.0. See `LICENSE` for details.
