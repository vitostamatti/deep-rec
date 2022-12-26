import tensorflow as tf
from tensorflow import keras
import numpy as np


from typing import List


class CategoricalFeatureEmbedding(keras.layers.Layer):
    """
    Transforms categorical features to tokens (embeddings).
    The module efficiently implements a collection of `keras.layers.Embedding` (with
    optional biases).
    Args:
        cardinalities: the number of distinct values for each feature. For example,
            :code:`cardinalities=[3, 4]` describes two features: the first one can
            take values in the range :code:`[0, 1, 2]` and the second one can take
            values in the range :code:`[0, 1, 2, 3]`.
        d_token: the size of one token.
        bias: if `True`, for each feature, a trainable vector is added to the
            embedding regardless of feature value. The bias vectors are not shared
            between features.
        initialization: initialization policy for parameters. Must be one of
            :code:`['uniform', 'normal']`. Let :code:`s = d ** -0.5`. Then, the
            corresponding distributions are :code:`Uniform(-s, s)` and :code:`Normal(0, s)`.
    """

    def __init__(
        self,
        cardinalities: List[int],
        emb_dim: int,
        use_bias: bool = True,
        initialization: str = 'uniform',
    ) -> None:

        super(CategoricalFeatureEmbedding, self).__init__()

        if not isinstance(cardinalities, list):
            raise ValueError("cardinalities must be a list of integers")

        if emb_dim < 0:
            raise ValueError('emb_dim must be positive')

        if not isinstance(emb_dim, int):
            raise ValueError('emb_dim must be positive integer')

        self.cardinalities = cardinalities
        self.emb_dim = emb_dim
        self.use_bias = use_bias
        self.initialization = initialization

        self.category_offsets = tf.cumsum(
            tf.constant([0] + cardinalities[:-1], dtype=tf.int64), axis=0
        )

        d_sqrt_inv = 1 / np.sqrt(emb_dim)

        if self.initialization == "uniform":
            self.initializer = tf.random_uniform_initializer(
                minval=-d_sqrt_inv,
                maxval=d_sqrt_inv,
                seed=None
            )
        elif self.initialization == "normal":
            self.initializer = tf.random_normal_initializer(
                mean=0.0, stddev=d_sqrt_inv, seed=None
            )

        self.embeddings = keras.layers.Embedding(
            input_dim=sum(cardinalities), output_dim=emb_dim,
            embeddings_initializer=self.initializer
        )

    def build(self, input_shape):
        self.bias = tf.Variable(
            initial_value=self.initializer(shape=(len(self.cardinalities), self.emb_dim),
                                           dtype=tf.float32),
            trainable=True) if self.use_bias else None

    def call(self, x):
        x = self.embeddings(x + self.category_offsets)
        if self.bias is not None:
            x = x + self.bias
        return x

    def get_config(self):
        config = super(CategoricalFeatureEmbedding, self).get_config()
        config.update({
            "cardinalities": self.cardinalities,
            "emb_dim": self.emb_dim,
            "use_bias": self.use_bias,
            "initialization": self.initialization,
        })
        return config


class MLP(keras.layers.Layer):
    """
    Multi-Layer Perceptron with Dropout layers
    and relu activations.

    Args:
        dims(list): list containing the number of units of each layer in the MLP.
        dropout(float): dropout percentage for each Dropout layer in between each Dense layer.
    """

    def __init__(self, dims: list, dropout=0.3):

        super(MLP, self).__init__()

        dims_pairs = list(zip(dims[:-1], dims[1:]))
        layers = []
        for idx, (dim_in, dim_out) in enumerate(dims_pairs):
            is_last = idx >= (len(dims_pairs) - 1)
            linear = keras.layers.Dense(units=dim_out, input_shape=(dim_in,))
            layers.append(linear)

            if is_last:
                continue

            layers.append(keras.layers.ReLU())

            # add dropout regularization
            dropout_layer = keras.layers.Dropout(dropout)
            layers.append(dropout_layer)

        self.mlp = keras.Sequential(layers)

    def call(self, inputs):
        return self.mlp(inputs)

    def get_config(self):
        config = super(MLP, self).get_config()
        config.update({
            "dims": self.dims,
        })
        return config
