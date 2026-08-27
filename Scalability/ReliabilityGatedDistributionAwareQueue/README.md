# Reliability-Gated Distribution-Aware Queue

This folder contains only the newest scalability model. It does not launch or
retrain RouteNet-Fermi, Global Weighted Queue, or the ungated
Distribution-Aware Queue model.

No data is generated or downloaded. The code uses the existing official
dataset under `RouteNet-Fermi/data/scalability`.

## Model change

The original sum channel is unchanged:

```text
s_q = sum_i x_qi
```

The auxiliary distribution encoding is multiplied by a deterministic
sample-reliability gate:

```text
n_eff = max(N_q - 1, 0)
g_q   = n_eff / (n_eff + tau)
M_q   = concat(s_q, g_q * e_q)
```

The default is `tau = 4`. Therefore `N_q = 0/1, 2, 3, 5` produce gates
`0, 0.2, 0.333, 0.5`, respectively. The gate depends on the number of incoming
flow messages at each queue, not on the total number of topology nodes.

## Five independent seeds

```text
1234, 2026, 3407, 5678, 9012
```

Every seed trains the newest model from scratch for 25 epochs and 2500 steps
per epoch. The validation rule, official test split, and 25 per-topology test
evaluations are unchanged. Only initialization and training shuffle change.

## RunPod workflow

Expected paths:

```text
/workspace/RouteNet-Fermi
/workspace/RouteNet-Fermi-WeightedQueue
```

First run the fast gate test and inspect the five generated commands:

```bash
cd /workspace/RouteNet-Fermi-WeightedQueue/Scalability/ReliabilityGatedDistributionAwareQueue
python smoke_test.py
python run_five_seeds.py \
  --source-root /workspace/RouteNet-Fermi \
  --python-bin python \
  --dry-run
```

Start all five seeds in the background:

```bash
bash start_five_seeds.sh
```

Check progress:

```bash
bash status_five_seeds.sh
```

Follow the complete log without stopping training:

```bash
tail -f "$(cat logs/latest_log.txt)"
```

Press `Ctrl+C` to leave `tail`. The background training continues.

## Safe restart

Run the same launcher again after an interruption:

```bash
bash start_five_seeds.sh
```

A trial is skipped only when all of the following are valid:

- `results.json` has the requested seed and newest aggregation label;
- `training_state.json` has status `complete`;
- `config.json` has the same reliability tau;
- `test_by_topology_size.csv` has all 25 topology sizes.

Incomplete attempts remain on disk. A restart creates a new timestamped
attempt, so no result is overwritten.

## Results

Each seed is saved under:

```text
runs/seed_<SEED>/attempt_<TIMESTAMP>/
```

After all requested seeds complete, the launcher automatically creates a new
timestamped summary under `summaries/`. It contains:

- overall MAPE for each seed;
- overall mean and standard deviation;
- per-topology MAPE for every seed;
- per-topology mean, standard deviation, minimum, and maximum;
- an overall five-seed figure;
- a full topology figure;
- a dedicated 50-to-100-node reliability figure.

The summary intentionally contains only the latest model.

## Local Windows checks

```powershell
.\run_task.ps1 smoke
.\run_task.ps1 dryrun
```

Use `formal` only when you intentionally want to perform all five full trials
on the current machine.

