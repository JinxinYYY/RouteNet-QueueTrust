"""Train Queue-message variants for the RouteNet-Fermi scalability experiment.

The official scalability dataset has train/test splits but no validation split.
This script deterministically reserves archives within every training topology
size, producing a small stratified validation set. The untouched test split is
evaluated only after reloading the checkpoint with the lowest validation MAPE.
"""

import argparse
import csv
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import tensorflow as tf

from weighted_queue_model import AGGREGATION_NAMES, RouteNet_Fermi


HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE_ROOT = HERE.parent.parent / "RouteNet-Fermi"
CACHE_VERSION = "v3_stratified_archive_split"

SCALABILITY_OUTPUT_SIGNATURE = (
    {
        "traffic": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "packets": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "length": tf.TensorSpec(shape=None, dtype=tf.int32),
        "model": tf.TensorSpec(shape=None, dtype=tf.int32),
        "eq_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "avg_pkts_lambda": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "exp_max_factor": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "pkts_lambda_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "avg_t_off": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "avg_t_on": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "ar_a": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "sigma": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "capacity": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "queue_size": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "policy": tf.TensorSpec(shape=None, dtype=tf.int32),
        "priority": tf.TensorSpec(shape=None, dtype=tf.int32),
        "weight": tf.TensorSpec(shape=(None, 1), dtype=tf.float32),
        "link_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
        "queue_to_path": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
        "queue_to_link": tf.RaggedTensorSpec(shape=(None, 1), dtype=tf.int32),
        "path_to_queue": tf.RaggedTensorSpec(
            shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
        ),
        "path_to_link": tf.RaggedTensorSpec(
            shape=(None, None, 2), dtype=tf.int32, ragged_rank=1
        ),
    },
    tf.TensorSpec(shape=None, dtype=tf.float32),
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Scalability Delay comparison with weighted Queue messages."
    )
    parser.add_argument(
        "--aggregation",
        choices=("sum", "sum_mlp", "weighted"),
        default="weighted",
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Defaults to SOURCE_ROOT/data/scalability.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Defaults to SOURCE_ROOT/scalability/delay/dataset_cache.",
    )
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--steps-per-epoch", type=int, default=2500)
    parser.add_argument("--validation-steps", type=int, default=None)
    parser.add_argument("--test-steps", type=int, default=None)
    parser.add_argument(
        "--fit-verbose",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Keras training output mode. 2 writes one line per epoch.",
    )
    parser.add_argument(
        "--evaluate-verbose",
        type=int,
        choices=(0, 1, 2),
        default=2,
        help="Keras test output mode. 2 avoids per-sample progress logging.",
    )
    parser.add_argument(
        "--skip-per-size-evaluation",
        action="store_true",
        help="Skip the additional 50-to-300-node test breakdown.",
    )
    parser.add_argument(
        "--validation-file-modulus",
        type=int,
        default=200,
        help="Within each training topology size, reserve every Nth archive.",
    )
    parser.add_argument("--shuffle-buffer", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--clipnorm", type=float, default=1.0)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--output-root", type=Path, default=HERE / "runs_scalability"
    )
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def load_data_module(source_root):
    scalability_code = source_root / "scalability" / "delay"
    generator_file = scalability_code / "data_generator.py"
    if not generator_file.is_file():
        raise FileNotFoundError(
            "Scalability data generator not found: {}".format(generator_file)
        )

    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(scalability_code))
    import data_generator

    return data_generator


def make_split_input_fn(data_module):
    def split_generator(data_dir, split, validation_file_modulus):
        if isinstance(data_dir, bytes):
            data_dir = data_dir.decode("UTF-8")
        if isinstance(split, bytes):
            split = split.decode("UTF-8")
        validation_file_modulus = int(validation_file_modulus)

        tool = data_module.DatanetAPI(
            data_dir, shuffle=(split == "train")
        )
        files_by_topology = {}
        for root, filename in tool.get_available_files():
            files_by_topology.setdefault(root, []).append((root, filename))

        selected_files = []
        for root in sorted(files_by_topology):
            topology_files = sorted(
                files_by_topology[root], key=lambda item: item[1]
            )
            for index, item in enumerate(topology_files):
                is_validation = index % validation_file_modulus == 0
                if (split == "validation" and is_validation) or (
                    split == "train" and not is_validation
                ):
                    selected_files.append(item)

        # The upstream setter contains a tuple-shadowing bug in this release;
        # assigning its documented internal selection list preserves the same
        # DatanetAPI iterator while keeping train/validation archives disjoint.
        tool._selected_tuple_files = selected_files

        for sample in tool:
            graph = data_module.nx.DiGraph(sample.get_topology_object())
            hypergraph = data_module.network_to_hypergraph(
                G=graph,
                R=sample.get_routing_matrix(),
                T=sample.get_traffic_matrix(),
                P=sample.get_performance_matrix(),
            )
            result = data_module.hypergraph_to_input_data(hypergraph)
            if all(value > 0 for value in result[1]):
                yield result

    def split_input_fn(data_dir, split, validation_file_modulus):
        dataset = tf.data.Dataset.from_generator(
            split_generator,
            args=[data_dir, split, validation_file_modulus],
            output_signature=SCALABILITY_OUTPUT_SIGNATURE,
        )
        return dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return split_input_fn


def make_datasets(
    split_input_fn,
    test_input_fn,
    data_root,
    cache_dir,
    cache_enabled,
    validation_file_modulus,
    validation_steps,
    test_steps,
    shuffle_buffer,
    seed,
):
    if validation_file_modulus < 2:
        raise ValueError("--validation-file-modulus must be at least 2")

    train_path = data_root / "train"
    test_path = data_root / "test"
    for path in (train_path, test_path):
        if not path.is_dir():
            raise FileNotFoundError("Dataset directory not found: {}".format(path))

    train = split_input_fn(
        str(train_path), "train", validation_file_modulus
    )
    validation = split_input_fn(
        str(train_path), "validation", validation_file_modulus
    )

    # The full training split is intentionally not disk-cached: 2,500 steps
    # do not exhaust it in one epoch, so tf.data would discard a partial cache.
    # Validation/test are finite and deterministic, so their converted graph
    # tensors can be reused by the second model without changing any values.
    if validation_steps is not None:
        validation = validation.take(validation_steps)

    if cache_enabled:
        cache_dir.mkdir(parents=True, exist_ok=True)
        validation_scope = (
            "full" if validation_steps is None else "steps{}".format(validation_steps)
        )
        validation_cache = cache_dir / "scalability_delay_validation_{}_m{}_{}".format(
            CACHE_VERSION, validation_file_modulus, validation_scope
        )
        print("Reusing scalability validation cache: {}".format(validation_cache))
        validation = validation.cache(str(validation_cache))

    train = train.shuffle(
        shuffle_buffer, seed=seed, reshuffle_each_iteration=True
    )
    train = train.repeat().prefetch(tf.data.experimental.AUTOTUNE)

    validation = validation.prefetch(tf.data.experimental.AUTOTUNE)

    test = test_input_fn(str(test_path), shuffle=False)
    if test_steps is not None:
        test = test.take(test_steps)
    if cache_enabled:
        test_scope = "full" if test_steps is None else "steps{}".format(test_steps)
        test_cache = cache_dir / "scalability_delay_test_{}_{}".format(
            CACHE_VERSION, test_scope
        )
        print("Reusing scalability test cache: {}".format(test_cache))
        test = test.cache(str(test_cache))
    test = test.prefetch(tf.data.experimental.AUTOTUNE)
    return train, validation, test


def check_finite_dataset(dataset, label, materialize_cache):
    """Check a finite dataset and optionally finish its on-disk cache.

    Iterating the whole validation stream once is intentional. TensorFlow only
    commits a file cache after reaching the end of the input; a one-sample
    probe would leave a partial cache that TensorFlow has to discard. The
    materialized tensors are byte-for-byte the same inputs later seen by fit().
    """
    if not materialize_cache:
        try:
            next(iter(dataset.take(1)))
        except StopIteration:
            raise ValueError("{} dataset is empty".format(label))
        return

    print("Materializing/reusing deterministic {} cache...".format(label))
    count = 0
    for count, _ in enumerate(dataset, start=1):
        if count % 100 == 0:
            print("  cached {} {} samples".format(label, count))
    if count == 0:
        raise ValueError("{} dataset is empty".format(label))
    print("{} cache ready: {} samples".format(label.capitalize(), count))


class AggregationWeightLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        del epoch
        weights = self.model.aggregation_weights()
        if weights is None:
            return
        logs = logs if logs is not None else {}
        values = {}
        for name in AGGREGATION_NAMES:
            value = float(weights[name].numpy())
            logs["alpha_{}".format(name)] = value
            values[name] = value
        print("\nQueue aggregation weights: {}".format(values))


class TrainingStateLogger(tf.keras.callbacks.Callback):
    """Persist completed-epoch state so an interrupted run remains auditable."""

    def __init__(self, output_file):
        super().__init__()
        self.output_file = Path(output_file)
        self.state = {
            "status": "created",
            "epochs_completed": 0,
            "started_at": None,
            "updated_at": None,
            "last_epoch_metrics": {},
        }

    @staticmethod
    def _timestamp():
        return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

    def _write(self):
        temporary = self.output_file.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as fp:
            json.dump(self.state, fp, indent=2, ensure_ascii=False)
        temporary.replace(self.output_file)

    def on_train_begin(self, logs=None):
        del logs
        self.state["status"] = "running"
        self.state["started_at"] = self._timestamp()
        self.state["updated_at"] = self.state["started_at"]
        self._write()

    def on_epoch_end(self, epoch, logs=None):
        metrics = {}
        for name, value in (logs or {}).items():
            try:
                metrics[name] = float(value)
            except (TypeError, ValueError):
                metrics[name] = str(value)
        self.state["epochs_completed"] = int(epoch) + 1
        self.state["last_epoch_metrics"] = metrics
        self.state["updated_at"] = self._timestamp()
        self._write()

    def on_train_end(self, logs=None):
        del logs
        self.state["status"] = "training_complete"
        self.state["updated_at"] = self._timestamp()
        self._write()

    def mark_test_complete(self, result):
        self.state["status"] = "complete"
        self.state["best_validation_mape"] = result["best_validation_mape"]
        self.state["test_mape"] = result["test_mape"]
        self.state["updated_at"] = self._timestamp()
        self._write()


def json_config(args, source_root, data_root, cache_dir, run_dir):
    result = vars(args).copy()
    result.update(
        {
            "source_root": str(source_root),
            "data_root": str(data_root),
            "cache_dir": str(cache_dir),
            "run_dir": str(run_dir),
            "validation_rule": "archive_index % {} == 0 within each topology size".format(
                args.validation_file_modulus
            ),
            "tensorflow_version": tf.__version__,
        }
    )
    result["output_root"] = str(args.output_root.resolve())
    return result


def evaluate_test_by_topology_size(model, input_fn, test_root, output_file):
    topology_dirs = [path for path in test_root.iterdir() if path.is_dir()]
    topology_dirs.sort(key=lambda path: int(path.name))
    rows = []
    for topology_dir in topology_dirs:
        dataset = input_fn(str(topology_dir), shuffle=False)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        metrics = model.evaluate(dataset, return_dict=True, verbose=0)
        row = {
            "topology_nodes": int(topology_dir.name),
            "test_mape": float(metrics["loss"]),
        }
        rows.append(row)
        print(
            "Topology {} nodes: test MAPE {:.6f}".format(
                row["topology_nodes"], row["test_mape"]
            )
        )

    with output_file.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(
            fp, fieldnames=("topology_nodes", "test_mape")
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    source_root = args.source_root.resolve()
    data_root = (
        args.data_root.resolve()
        if args.data_root is not None
        else source_root / "data" / "scalability"
    )
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else source_root / "scalability" / "delay" / "dataset_cache"
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.run_name or "{}_{}ep_{}".format(
        args.aggregation, args.epochs, timestamp
    )
    run_dir = args.output_root.resolve() / run_name
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            "Run directory is not empty; no files were overwritten: {}".format(
                run_dir
            )
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with (run_dir / "config.json").open("w", encoding="utf-8") as fp:
        json.dump(
            json_config(args, source_root, data_root, cache_dir, run_dir),
            fp,
            indent=2,
            ensure_ascii=False,
        )

    data_module = load_data_module(source_root)
    split_input_fn = make_split_input_fn(data_module)
    test_input_fn = data_module.input_fn
    train, validation, test = make_datasets(
        split_input_fn=split_input_fn,
        test_input_fn=test_input_fn,
        data_root=data_root,
        cache_dir=cache_dir,
        cache_enabled=not args.no_cache,
        validation_file_modulus=args.validation_file_modulus,
        validation_steps=args.validation_steps,
        test_steps=args.test_steps,
        shuffle_buffer=args.shuffle_buffer,
        seed=args.seed,
    )

    # Confirm that both parts of the deterministic training split are non-empty.
    sample_inputs, _ = next(iter(train.take(1)))
    check_finite_dataset(
        validation,
        "validation",
        materialize_cache=not args.no_cache,
    )

    model = RouteNet_Fermi(queue_aggregation=args.aggregation)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=args.learning_rate, clipnorm=args.clipnorm
    )
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanAbsolutePercentageError(),
        run_eagerly=False,
    )
    model(sample_inputs)
    print("Model parameters: {:,}".format(model.count_params()))
    print("Validation split: every {}th archive within each training topology".format(
        args.validation_file_modulus
    ))

    best_prefix = checkpoint_dir / "best"
    last_prefix = checkpoint_dir / "last"
    state_logger = TrainingStateLogger(run_dir / "training_state.json")
    callbacks = [
        AggregationWeightLogger(),
        tf.keras.callbacks.CSVLogger(
            str(run_dir / "history.csv"), append=False
        ),
        state_logger,
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_prefix),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            save_weights_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(last_prefix),
            save_best_only=False,
            save_weights_only=True,
            verbose=0,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
            min_lr=1.0e-5,
            verbose=1,
        ),
    ]

    history = model.fit(
        train,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        validation_data=validation,
        validation_steps=args.validation_steps,
        callbacks=callbacks,
        verbose=args.fit_verbose,
    )

    if not Path(str(best_prefix) + ".index").is_file():
        raise FileNotFoundError("Best validation checkpoint was not created")
    model.load_weights(str(best_prefix))
    test_metrics = model.evaluate(
        test,
        steps=args.test_steps,
        return_dict=True,
        verbose=args.evaluate_verbose,
    )

    per_size_results = None
    if not args.skip_per_size_evaluation and args.test_steps is None:
        per_size_results = evaluate_test_by_topology_size(
            model,
            test_input_fn,
            data_root / "test",
            run_dir / "test_by_topology_size.csv",
        )

    learned_weights = model.aggregation_weights()
    learned_weights_json = None
    if learned_weights is not None:
        learned_weights_json = {
            name: float(learned_weights[name].numpy())
            for name in AGGREGATION_NAMES
        }
        with (run_dir / "learned_queue_weights.json").open(
            "w", encoding="utf-8"
        ) as fp:
            json.dump(learned_weights_json, fp, indent=2)

    result = {
        "experiment": "scalability_delay",
        "aggregation": args.aggregation,
        "seed": args.seed,
        "parameters": model.count_params(),
        "epochs_completed": len(history.history.get("loss", [])),
        "validation_file_modulus": args.validation_file_modulus,
        "best_validation_mape": min(history.history["val_loss"]),
        "test_mape": float(test_metrics["loss"]),
        "test_topology_sizes": (
            len(per_size_results) if per_size_results is not None else None
        ),
        "learned_queue_weights": learned_weights_json,
    }
    with (run_dir / "results.json").open("w", encoding="utf-8") as fp:
        json.dump(result, fp, indent=2)
    state_logger.mark_test_complete(result)

    print("Scalability run artifacts: {}".format(run_dir))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
