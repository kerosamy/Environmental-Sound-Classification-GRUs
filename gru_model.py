import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    GRU, Dense, Dropout, BatchNormalization,
    Bidirectional, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


class GRUClassifier:
    def __init__(
        self,
        input_shape,
        num_classes=10,
        gru_units=(128, 64),
        dropout_rate=0.4,
        recurrent_dropout=0.2,
        learning_rate=5e-4,
        l2_reg=1e-4,
    ):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.gru_units = gru_units
        self.dropout_rate = dropout_rate
        self.recurrent_dropout = recurrent_dropout
        self.learning_rate = learning_rate
        self.l2_reg = l2_reg

        self.model = self._build()

    def _build(self):
        inp = Input(shape=self.input_shape)

        x = inp

        # ── GRU STACK ─────────────────────────────
        for i, units in enumerate(self.gru_units):
            return_seq = i != len(self.gru_units) - 1  # last GRU → no sequence

            x = Bidirectional(
                GRU(
                    units,
                    return_sequences=return_seq,
                    dropout=self.dropout_rate,
                    recurrent_dropout=self.recurrent_dropout,
                    kernel_regularizer=l2(self.l2_reg),
                )
            )(x)

            if return_seq:
                x = BatchNormalization()(x)

        # ── DENSE HEAD ─────────────────────────────
        x = Dense(128, activation="relu", kernel_regularizer=l2(self.l2_reg))(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(64, activation="relu")(x)
        x = Dropout(self.dropout_rate / 2)(x)

        out = Dense(self.num_classes, activation="softmax")(x)

        model = Model(inp, out)

        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate, clipnorm=1.0),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        return model

    def summary(self):
        self.model.summary()