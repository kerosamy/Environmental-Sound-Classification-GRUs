import numpy as np
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, BatchNormalization,
    Bidirectional, Input, Layer, Conv1D, MaxPooling1D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class AttentionLayer(Layer):
    """Soft attention over time steps."""

    def build(self, input_shape):
        self.W = self.add_weight(
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True,
            name="attn_weight"
        )
        super().build(input_shape)

    def call(self, x):
        score  = tf.nn.tanh(tf.matmul(x, self.W))   # (batch, T, 1)
        weight = tf.nn.softmax(score, axis=1)         # (batch, T, 1)
        return tf.reduce_sum(x * weight, axis=1)      # (batch, units)


class GRUClassifier:
    """
    Improved GRU classifier:
      Conv1D  →  Bidirectional GRU stack  →  Attention  →  Dense head
    """

    def __init__(
        self,
        input_shape,
        num_classes    = 10,
        gru_units      = [128, 64],
        dense_units    = 128,
        dropout_rate   = 0.4,
        recurrent_drop = 0.2,
        learning_rate  = 5e-4,
        l2_reg         = 1e-4,
    ):
        self.input_shape   = input_shape
        self.num_classes   = num_classes
        self.gru_units     = gru_units
        self.dense_units   = dense_units
        self.dropout_rate  = dropout_rate
        self.recurrent_drop= recurrent_drop
        self.learning_rate = learning_rate
        self.l2_reg        = l2_reg

        self.model = self._build()

    def _build(self) -> Model:
        inp = Input(shape=self.input_shape)

        # ── 1. Conv1D front-end ──────────────────────────────────
        # Extracts local temporal patterns before the GRU sees them.
        # Reduces sequence length → faster + better gradient flow.
        x = Conv1D(64, kernel_size=3, padding="same", activation="relu")(inp)
        x = BatchNormalization()(x)
        x = MaxPooling1D(pool_size=2)(x)          # halves sequence length
        x = Dropout(self.dropout_rate)(x)

        # ── 2. Bidirectional GRU stack ───────────────────────────
        for i, units in enumerate(self.gru_units):
            return_seq = True   # always True — Attention will collapse time

            x = Bidirectional(GRU(
                units,
                return_sequences  = return_seq,
                dropout           = self.dropout_rate,
                recurrent_dropout = self.recurrent_drop,
                kernel_regularizer= l2(self.l2_reg),
            ))(x)
            x = BatchNormalization()(x)

        # ── 3. Attention ─────────────────────────────────────────
        x = AttentionLayer()(x)                   # (batch, units*2) after Bidirectional

        # ── 4. Dense head ────────────────────────────────────────
        x = Dense(self.dense_units, activation="relu",
                  kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(self.dense_units // 2, activation="relu")(x)
        x = Dropout(self.dropout_rate / 2)(x)

        out = Dense(self.num_classes, activation="softmax")(x)

        # ── 5. Compile ───────────────────────────────────────────
        model = Model(inp, out)
        model.compile(
            optimizer = Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss      = "sparse_categorical_crossentropy",
            metrics   = ["accuracy"]
        )

        return model

    def summary(self):
        self.model.summary()

    def get_name(self) -> str:
        units_str = "_".join(str(u) for u in self.gru_units)
        return f"BiGRU_Attn_{units_str}_dr{self.dropout_rate}_lr{self.learning_rate}"