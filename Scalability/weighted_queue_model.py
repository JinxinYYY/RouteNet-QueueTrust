"""RouteNet-Fermi with learnable multi-statistic Path-to-Queue messages.

The backbone follows the official ``delay_model.py`` implementation.  The only
algorithmic change is the Path-to-Queue aggregation stage.  Three modes are
provided so that the change can be studied without touching the dataset:

``sum``
    The original RouteNet-Fermi sum aggregation.
``sum_mlp``
    Sum followed by the same type of bounded projection used by each branch.
``weighted``
    Sum, mean, max and log-count are projected independently and combined with
    four trainable global Softmax weights.
"""

import tensorflow as tf


AGGREGATION_NAMES = ("sum", "mean", "max", "count")


class MessageProjection(tf.keras.layers.Layer):
    """Maps one aggregation statistic to a bounded queue-message vector."""

    def __init__(self, output_dim, name=None):
        super().__init__(name=name)
        self.hidden = tf.keras.layers.Dense(output_dim, activation="relu")
        # Tanh puts all four branches on the same bounded numerical scale while
        # retaining magnitude information (especially important for count).
        self.output_layer = tf.keras.layers.Dense(output_dim, activation="tanh")

    def call(self, inputs):
        return self.output_layer(self.hidden(inputs))


class QueueMessageAggregator(tf.keras.layers.Layer):
    """Aggregates all path messages that arrive at each queue."""

    def __init__(self, state_dim, mode="weighted", name="queue_message_aggregator"):
        super().__init__(name=name)
        if mode not in ("sum_mlp", "weighted"):
            raise ValueError("QueueMessageAggregator mode must be sum_mlp or weighted")

        self.state_dim = state_dim
        self.mode = mode
        self.sum_projection = MessageProjection(state_dim, name="sum_projection")

        if mode == "weighted":
            self.mean_projection = MessageProjection(state_dim, name="mean_projection")
            self.max_projection = MessageProjection(state_dim, name="max_projection")
            self.count_projection = MessageProjection(state_dim, name="count_projection")
            # Equal initial contribution: softmax([0, 0, 0, 0]) = 0.25 each.
            self.aggregation_logits = self.add_weight(
                name="aggregation_logits",
                shape=(len(AGGREGATION_NAMES),),
                initializer="zeros",
                trainable=True,
            )

    def _statistics(self, path_messages):
        if not isinstance(path_messages, tf.RaggedTensor):
            raise TypeError("path_messages must be a RaggedTensor")

        dtype = path_messages.dtype
        counts = tf.cast(path_messages.row_lengths(), dtype)
        safe_counts = tf.maximum(counts, tf.ones_like(counts))

        message_sum = tf.reduce_sum(path_messages, axis=1)
        message_mean = message_sum / tf.expand_dims(safe_counts, axis=-1)

        # Ragged reduce_max avoids padding all queue-flow messages into a dense
        # tensor, which is important when evaluating larger topologies.
        message_max = tf.reduce_max(path_messages, axis=1)
        message_max = tf.where(
            tf.expand_dims(counts > 0, axis=-1),
            message_max,
            tf.zeros_like(message_max),
        )
        message_count = tf.expand_dims(tf.math.log1p(counts), axis=-1)

        return message_sum, message_mean, message_max, message_count

    def call(self, path_messages):
        message_sum, message_mean, message_max, message_count = self._statistics(
            path_messages
        )

        projected_sum = self.sum_projection(message_sum)
        if self.mode == "sum_mlp":
            return projected_sum

        projected_messages = tf.stack(
            [
                projected_sum,
                self.mean_projection(message_mean),
                self.max_projection(message_max),
                self.count_projection(message_count),
            ],
            axis=1,
        )
        weights = tf.nn.softmax(self.aggregation_logits)
        return tf.reduce_sum(
            projected_messages * tf.reshape(weights, (1, -1, 1)), axis=1
        )

    def normalized_weights(self):
        if self.mode != "weighted":
            raise ValueError("Learnable aggregation weights exist only in weighted mode")
        return tf.nn.softmax(self.aggregation_logits)


class RouteNet_Fermi(tf.keras.Model):
    """Official RouteNet-Fermi delay model with selectable queue aggregation."""

    def __init__(self, queue_aggregation="weighted"):
        super().__init__()
        if queue_aggregation not in ("sum", "sum_mlp", "weighted"):
            raise ValueError("queue_aggregation must be sum, sum_mlp or weighted")

        self.queue_aggregation = queue_aggregation
        self.max_num_models = 7
        self.num_policies = 4
        self.max_num_queues = 3

        self.iterations = 8
        self.path_state_dim = 32
        self.link_state_dim = 32
        self.queue_state_dim = 32

        self.z_score = {
            "traffic": [1385.4058837890625, 859.8118896484375],
            "packets": [1.4015231132507324, 0.8932565450668335],
            "eq_lambda": [1350.97119140625, 858.316162109375],
            "avg_pkts_lambda": [0.9117304086685181, 0.9723503589630127],
            "exp_max_factor": [6.663637638092041, 4.715115070343018],
            "pkts_lambda_on": [0.9116322994232178, 1.651275396347046],
            "avg_t_off": [1.6649284362792969, 2.356407403945923],
            "avg_t_on": [1.6649284362792969, 2.356407403945923],
            "ar_a": [0.0, 1.0],
            "sigma": [0.0, 1.0],
            "capacity": [27611.091796875, 20090.62109375],
            "queue_size": [30259.10546875, 21410.095703125],
        }

        self.path_update = tf.keras.layers.GRUCell(self.path_state_dim)
        self.link_update = tf.keras.layers.GRUCell(self.link_state_dim)
        self.queue_update = tf.keras.layers.GRUCell(self.queue_state_dim)

        if queue_aggregation == "sum":
            self.queue_message_aggregator = None
        else:
            self.queue_message_aggregator = QueueMessageAggregator(
                self.queue_state_dim, mode=queue_aggregation
            )

        self.path_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=10 + self.max_num_models),
                tf.keras.layers.Dense(self.path_state_dim, activation="relu"),
                tf.keras.layers.Dense(self.path_state_dim, activation="relu"),
            ],
            name="path_embedding",
        )

        self.queue_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.max_num_queues + 2),
                tf.keras.layers.Dense(self.queue_state_dim, activation="relu"),
                tf.keras.layers.Dense(self.queue_state_dim, activation="relu"),
            ],
            name="queue_embedding",
        )

        self.link_embedding = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=self.num_policies + 1),
                tf.keras.layers.Dense(self.link_state_dim, activation="relu"),
                tf.keras.layers.Dense(self.link_state_dim, activation="relu"),
            ],
            name="link_embedding",
        )

        self.readout_path = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(None, self.path_state_dim)),
                tf.keras.layers.Dense(int(self.link_state_dim / 2), activation="relu"),
                tf.keras.layers.Dense(int(self.path_state_dim / 2), activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="PathReadout",
        )

    def aggregation_weights(self):
        """Returns a name-to-weight mapping tensor for the weighted model."""
        if self.queue_aggregation != "weighted":
            return None
        weights = self.queue_message_aggregator.normalized_weights()
        return dict(zip(AGGREGATION_NAMES, tf.unstack(weights)))

    @tf.function
    def call(self, inputs):
        traffic = inputs["traffic"]
        packets = inputs["packets"]
        length = inputs["length"]
        model = inputs["model"]
        eq_lambda = inputs["eq_lambda"]
        avg_pkts_lambda = inputs["avg_pkts_lambda"]
        exp_max_factor = inputs["exp_max_factor"]
        pkts_lambda_on = inputs["pkts_lambda_on"]
        avg_t_off = inputs["avg_t_off"]
        avg_t_on = inputs["avg_t_on"]
        ar_a = inputs["ar_a"]
        sigma = inputs["sigma"]

        capacity = inputs["capacity"]
        policy = tf.one_hot(inputs["policy"], self.num_policies)

        queue_size = inputs["queue_size"]
        priority = tf.one_hot(inputs["priority"], self.max_num_queues)
        weight = inputs["weight"]

        queue_to_path = inputs["queue_to_path"]
        link_to_path = inputs["link_to_path"]
        path_to_link = inputs["path_to_link"]
        path_to_queue = inputs["path_to_queue"]
        queue_to_link = inputs["queue_to_link"]

        path_gather_traffic = tf.gather(traffic, path_to_link[:, :, 0])
        load = tf.reduce_sum(path_gather_traffic, axis=1) / capacity
        pkt_size = traffic / packets

        path_state = self.path_embedding(
            tf.concat(
                [
                    (traffic - self.z_score["traffic"][0])
                    / self.z_score["traffic"][1],
                    (packets - self.z_score["packets"][0])
                    / self.z_score["packets"][1],
                    tf.one_hot(model, self.max_num_models),
                    (eq_lambda - self.z_score["eq_lambda"][0])
                    / self.z_score["eq_lambda"][1],
                    (avg_pkts_lambda - self.z_score["avg_pkts_lambda"][0])
                    / self.z_score["avg_pkts_lambda"][1],
                    (exp_max_factor - self.z_score["exp_max_factor"][0])
                    / self.z_score["exp_max_factor"][1],
                    (pkts_lambda_on - self.z_score["pkts_lambda_on"][0])
                    / self.z_score["pkts_lambda_on"][1],
                    (avg_t_off - self.z_score["avg_t_off"][0])
                    / self.z_score["avg_t_off"][1],
                    (avg_t_on - self.z_score["avg_t_on"][0])
                    / self.z_score["avg_t_on"][1],
                    (ar_a - self.z_score["ar_a"][0]) / self.z_score["ar_a"][1],
                    (sigma - self.z_score["sigma"][0])
                    / self.z_score["sigma"][1],
                ],
                axis=1,
            )
        )

        link_state = self.link_embedding(tf.concat([load, policy], axis=1))
        queue_state = self.queue_embedding(
            tf.concat(
                [
                    (queue_size - self.z_score["queue_size"][0])
                    / self.z_score["queue_size"][1],
                    priority,
                    weight,
                ],
                axis=1,
            )
        )

        for _ in range(self.iterations):
            queue_gather = tf.gather(queue_state, queue_to_path)
            link_gather = tf.gather(link_state, link_to_path, name="LinkToPath")
            path_update_rnn = tf.keras.layers.RNN(
                self.path_update, return_sequences=True, return_state=True
            )
            previous_path_state = path_state
            path_state_sequence, path_state = path_update_rnn(
                tf.concat([queue_gather, link_gather], axis=2),
                initial_state=path_state,
            )
            path_state_sequence = tf.concat(
                [tf.expand_dims(previous_path_state, 1), path_state_sequence], axis=1
            )

            path_gather = tf.gather_nd(path_state_sequence, path_to_queue)
            if self.queue_aggregation == "sum":
                queue_message = tf.reduce_sum(path_gather, axis=1)
            else:
                queue_message = self.queue_message_aggregator(path_gather)
            queue_state, _ = self.queue_update(queue_message, [queue_state])

            queue_gather = tf.gather(queue_state, queue_to_link)
            link_gru_rnn = tf.keras.layers.RNN(
                self.link_update, return_sequences=False
            )
            link_state = link_gru_rnn(queue_gather, initial_state=link_state)

        capacity_gather = tf.gather(capacity, link_to_path)
        input_tensor = path_state_sequence[:, 1:].to_tensor()
        occupancy_gather = self.readout_path(input_tensor)
        length = tf.ensure_shape(length, [None])
        occupancy_gather = tf.RaggedTensor.from_tensor(
            occupancy_gather, lengths=length
        )

        queue_delay = tf.reduce_sum(occupancy_gather / capacity_gather, axis=1)
        trans_delay = pkt_size * tf.reduce_sum(1 / capacity_gather, axis=1)
        return queue_delay + trans_delay
