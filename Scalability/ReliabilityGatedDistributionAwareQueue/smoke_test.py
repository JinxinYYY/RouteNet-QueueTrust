"""Fast unit smoke test for the reliability gate and output shape."""

import numpy as np
import tensorflow as tf

from reliability_gated_model import ReliabilityGatedDistributionEncoder


def main():
    tf.random.set_seed(1234)
    flat_values = tf.constant(
        [
            [1.0, 2.0],
            [1.0, 3.0],
            [3.0, 5.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
            [5.0, 5.0],
        ],
        dtype=tf.float32,
    )
    messages = tf.RaggedTensor.from_row_lengths(flat_values, [0, 1, 2, 5])
    encoder = ReliabilityGatedDistributionEncoder(
        state_dim=2, hidden_dim=4, reliability_tau=4.0
    )
    queue_message, _, statistics = encoder.diagnostics(messages)

    expected_gate = np.asarray([0.0, 0.0, 0.2, 0.5], dtype=np.float32)
    np.testing.assert_allclose(
        statistics["reliability_gate"].numpy().reshape(-1),
        expected_gate,
        rtol=1.0e-6,
        atol=1.0e-6,
    )
    if tuple(queue_message.shape) != (4, 4):
        raise AssertionError("Unexpected queue-message shape: {}".format(queue_message.shape))
    np.testing.assert_allclose(
        queue_message[:, :2].numpy(), statistics["sum"].numpy(), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        statistics["gated_distribution"].numpy()[:2],
        np.zeros((2, 2), dtype=np.float32),
        rtol=0.0,
        atol=1.0e-7,
    )
    print("Reliability gates: {}".format(expected_gate.tolist()))
    print("Queue-message shape: {}".format(tuple(queue_message.shape)))
    print("Smoke test passed.")


if __name__ == "__main__":
    main()

