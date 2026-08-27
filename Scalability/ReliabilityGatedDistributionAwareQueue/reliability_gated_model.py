"""Reliability-gated distribution-aware Queue encoder for scalability.

The original sum message is preserved exactly.  Only the auxiliary
distribution encoding is attenuated when a queue contains too few incoming
flow messages for its empirical statistics to be reliable:

    gate(N) = max(N - 1, 0) / (max(N - 1, 0) + tau)
    queue_message = concat(message_sum, gate(N) * distribution_encoding)

This module is compatible with Python 3.6 and TensorFlow 2.6.
"""

import sys
from pathlib import Path

import tensorflow as tf


SCALABILITY_ROOT = Path(__file__).resolve().parent.parent
if str(SCALABILITY_ROOT) not in sys.path:
    sys.path.insert(0, str(SCALABILITY_ROOT))

from weighted_queue_model import RouteNet_Fermi  # noqa: E402


DIAGNOSTIC_NAMES = (
    "relative_count",
    "log_variance",
    "standardized_max",
    "reliability_gate",
    "distribution_encoding_rms",
    "gated_distribution_rms",
    "sum_message_rms",
)


class ReliabilityGatedDistributionEncoder(tf.keras.layers.Layer):
    """Encode Queue-message distributions with a deterministic count gate."""

    def __init__(
        self,
        state_dim=32,
        hidden_dim=32,
        reliability_tau=4.0,
        epsilon=1.0e-5,
        statistic_clip=8.0,
        name="reliability_gated_distribution_encoder",
    ):
        super().__init__(name=name)
        self.state_dim = int(state_dim)
        self.hidden_dim = int(hidden_dim)
        self.reliability_tau = float(reliability_tau)
        self.epsilon = float(epsilon)
        self.statistic_clip = float(statistic_clip)
        if self.reliability_tau <= 0.0:
            raise ValueError("reliability_tau must be positive")

        self.distribution_encoder = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=3 * self.state_dim + 1),
                tf.keras.layers.Dense(
                    self.hidden_dim,
                    activation="relu",
                    name="distribution_hidden",
                ),
                tf.keras.layers.Dense(
                    self.state_dim,
                    activation="tanh",
                    name="distribution_output",
                ),
            ],
            name="distribution_encoder",
        )
        self.diagnostic_metrics = {
            metric_name: tf.keras.metrics.Mean(name=metric_name)
            for metric_name in DIAGNOSTIC_NAMES
        }

    def _statistics(self, path_messages):
        if not isinstance(path_messages, tf.RaggedTensor):
            raise TypeError("path_messages must be a RaggedTensor")

        dtype = path_messages.dtype
        counts = tf.cast(path_messages.row_lengths(), dtype)
        safe_counts = tf.maximum(counts, tf.ones_like(counts))
        expanded_counts = tf.expand_dims(safe_counts, axis=-1)

        message_sum = tf.reduce_sum(path_messages, axis=1)
        message_mean = message_sum / expanded_counts

        row_ids = path_messages.value_rowids()
        centered_flat = path_messages.flat_values - tf.gather(message_mean, row_ids)
        squared = tf.RaggedTensor.from_row_splits(
            tf.square(centered_flat), path_messages.row_splits, validate=False
        )
        message_variance = tf.reduce_sum(squared, axis=1) / expanded_counts
        message_variance = tf.where(
            tf.expand_dims(counts > 0, axis=-1),
            message_variance,
            tf.zeros_like(message_variance),
        )

        message_max = tf.reduce_max(path_messages, axis=1)
        message_max = tf.where(
            tf.expand_dims(counts > 0, axis=-1),
            message_max,
            tf.zeros_like(message_max),
        )
        standardized_max = (message_max - message_mean) / tf.sqrt(
            message_variance + self.epsilon
        )
        standardized_max = tf.where(
            tf.expand_dims(counts > 1, axis=-1),
            standardized_max,
            tf.zeros_like(standardized_max),
        )
        standardized_max = tf.clip_by_value(
            standardized_max, -self.statistic_clip, self.statistic_clip
        )

        log_variance = tf.clip_by_value(
            tf.math.log1p(message_variance), 0.0, self.statistic_clip
        )

        log_count = tf.math.log1p(counts)
        graph_mean_log_count = tf.reduce_mean(log_count)
        relative_count = log_count / (graph_mean_log_count + self.epsilon)
        relative_count = tf.expand_dims(
            tf.clip_by_value(relative_count, 0.0, self.statistic_clip), axis=-1
        )

        effective_count = tf.maximum(counts - 1.0, tf.zeros_like(counts))
        reliability_gate = effective_count / (
            effective_count + tf.cast(self.reliability_tau, dtype)
        )
        reliability_gate = tf.expand_dims(reliability_gate, axis=-1)

        return (
            counts,
            message_sum,
            message_mean,
            log_variance,
            standardized_max,
            relative_count,
            reliability_gate,
        )

    def _compute(self, path_messages):
        (
            counts,
            message_sum,
            message_mean,
            log_variance,
            standardized_max,
            relative_count,
            reliability_gate,
        ) = self._statistics(path_messages)

        distribution_features = tf.concat(
            [message_mean, log_variance, standardized_max, relative_count], axis=-1
        )
        distribution_encoding = self.distribution_encoder(distribution_features)
        gated_distribution = reliability_gate * distribution_encoding
        queue_message = tf.concat([message_sum, gated_distribution], axis=-1)

        sum_rms = tf.sqrt(
            tf.reduce_mean(tf.square(message_sum), axis=-1) + self.epsilon
        )
        distribution_rms = tf.sqrt(
            tf.reduce_mean(tf.square(distribution_encoding), axis=-1) + self.epsilon
        )
        gated_distribution_rms = tf.sqrt(
            tf.reduce_mean(tf.square(gated_distribution), axis=-1) + self.epsilon
        )
        diagnostics = {
            "relative_count": tf.reduce_mean(relative_count),
            "log_variance": tf.reduce_mean(log_variance),
            "standardized_max": tf.reduce_mean(tf.abs(standardized_max)),
            "reliability_gate": tf.reduce_mean(reliability_gate),
            "distribution_encoding_rms": tf.reduce_mean(distribution_rms),
            "gated_distribution_rms": tf.reduce_mean(gated_distribution_rms),
            "sum_message_rms": tf.reduce_mean(sum_rms),
        }
        statistics = {
            "count": counts,
            "sum": message_sum,
            "mean": message_mean,
            "log_variance": log_variance,
            "standardized_max": standardized_max,
            "relative_count": relative_count,
            "reliability_gate": reliability_gate,
            "distribution_encoding": distribution_encoding,
            "gated_distribution": gated_distribution,
        }
        return queue_message, diagnostics, statistics

    def call(self, path_messages):
        queue_message, diagnostics, _ = self._compute(path_messages)
        for metric_name, value in diagnostics.items():
            self.diagnostic_metrics[metric_name].update_state(value)
        return queue_message

    def diagnostics(self, path_messages):
        """Return messages and gate diagnostics without changing model design."""

        return self._compute(path_messages)


class RouteNet_Fermi_ReliabilityGatedDistributionAware(RouteNet_Fermi):
    """RouteNet-Fermi with sum-preserving reliability-gated Queue encoding."""

    def __init__(
        self,
        encoder_hidden_dim=32,
        reliability_tau=4.0,
        queue_aggregation=None,
    ):
        del queue_aggregation
        super().__init__(queue_aggregation="sum_mlp")
        self.queue_aggregation = "reliability_gated_distribution_aware"
        self.reliability_tau = float(reliability_tau)
        self.queue_message_aggregator = ReliabilityGatedDistributionEncoder(
            state_dim=self.queue_state_dim,
            hidden_dim=encoder_hidden_dim,
            reliability_tau=self.reliability_tau,
        )

    def aggregation_weights(self):
        """Expose read-only diagnostics through the preserved training logger."""
        return {
            name: self.queue_message_aggregator.diagnostic_metrics[name].result()
            for name in DIAGNOSTIC_NAMES
        }
