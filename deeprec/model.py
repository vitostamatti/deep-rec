import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from typing import (Dict, List, Union)

from .layers import (
    CategoricalFeatureEmbedding,
    MLP
)


class RecModel:
    def __init__(
        self,
        emb_dim: int = 32,
        users_dim: int = 8,
        items_dim: int = 8,
        use_bias: bool = True,
        mlp_dims = [],
        dropout = 0.3
    ):

        self.emb_dim = emb_dim
        self.use_bias = use_bias
        self.users_dim = users_dim
        self.items_dim = items_dim
        self.mlp_dims = mlp_dims
        self.dropout = dropout

    def build(
        self,
        users: pd.Series,
        items: pd.Series,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None
    ):

        # build interaction model
        inter_model = build_interactions_model(
            users, items, self.emb_dim, self.use_bias
        )

        # build users model
        users_model = build_features_model(
            user_features,
            emb_dim=self.users_dim,
            use_bias=self.use_bias
        )

        # build items model
        items_model = build_features_model(
            item_features,
            emb_dim=self.users_dim,
            use_bias=self.use_bias
        )

        # build full model
        self.model = build_full_model(
            inter_model,
            users_model,
            items_model,
            self.mlp_dims,
            self.dropout
        )

    def __call__(self, X: Dict[str, np.ndarray]):
        return self.model(X)

    def fit(
        self,
        X: Dict[str, np.ndarray],
        y: np.ndarray,
        val_size=0.1,
        learning_rate=1e-4,
        epochs=20,
        batch_size=256,
        verbose=2,
        patience=3
    ):

        optimizer = keras.optimizers.Adam(
            learning_rate=learning_rate
        )

        self.model.compile(loss='mean_squared_error', optimizer=optimizer)

        early_stop = keras.callbacks.EarlyStopping(
            monitor='loss', patience=patience)

        X_train, y_train, X_val, y_val = split_data(X, y, val_size)

        history = self.model.fit(
            x=X_train,
            y=y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            validation_data=(X_val, y_val),
            callbacks=[early_stop],
        )

        return history

    def predict(self, X: Dict[str, np.ndarray]):
        return self.model.predict(X)

    def evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray):
        return self.model.evaluate(X, y)


def get_input_layers(X: pd.DataFrame):
    inputs = []
    for name in X.columns:
        x = X[name]
        input_layer = keras.Input(shape=(1,), name=name, dtype=x.dtype)
        inputs.append(input_layer)
    return inputs


def get_lookup_layer(x: pd.Series, name=None):
    name = x.name if name == None else name
    if name == None:
        raise ValueError(
            "If no name is provided, input series must have a name.")
    if x.dtypes == int:
        enc_layer = keras.layers.IntegerLookup(
            max_tokens=x.nunique(), name=f"encoding_{name}"
        )
    elif x.dtypes == "object":
        enc_layer = keras.layers.StringLookup(
            max_tokens=x.nunique(), name=f"encoding_{name}")
    else:
        raise ValueError("Data type must be either integer or string")

    enc_layer.adapt(x)

    return enc_layer


def get_encoding_layers(X: pd.DataFrame):
    enc_layers = []
    for name in X.columns:
        x = X[name]
        enc_layer = get_lookup_layer(x)
        enc_layers.append(enc_layer)
    return enc_layers


def build_interactions_model(users: pd.Series, items: pd.Series, emb_dim: int, use_bias=True):

    n_users, n_items = users.nunique(), items.nunique()

    users_input = keras.layers.Input(shape=(1,), name=users.name)

    users_enc_layer = tf.keras.layers.IntegerLookup(
        max_tokens=n_users, name="encoding_users"
    )

    users_enc_layer.adapt(users)

    users_enc = users_enc_layer(users_input)

    users_embedding = keras.layers.Embedding(
        input_dim=n_users,
        output_dim=emb_dim,
        input_length=1,
        name='users_embedding'
    )(users_enc)

    users_flatten = keras.layers.Flatten(
        name='users_flatten')(users_embedding)

    items_input = keras.layers.Input(shape=(1,), name=items.name)

    items_enc_layer = tf.keras.layers.IntegerLookup(
        max_tokens=n_items, name="encoding_items"
    )

    items_enc_layer.adapt(items)

    items_enc = items_enc_layer(items_input)

    items_embedding = keras.layers.Embedding(
        input_dim=n_items,
        output_dim=emb_dim,
        input_length=1,
        name='items_embedding'
    )(items_enc)

    items_flatten = keras.layers.Flatten(
        name='items_flatten')(items_embedding)

    output = keras.layers.Dot(name='dot_product', axes=1)(
        [users_flatten, items_flatten]
    )

    model = keras.Model([users_input, items_input], output)

    return model


def build_features_model(X: pd.DataFrame, emb_dim=32, use_bias=True):

    inputs = get_input_layers(X)
    enc_layers = get_encoding_layers(X)

    x = []
    cardinalities = []
    for idx, layer in enumerate(inputs):
        cardinalities.append(len(enc_layers[idx].get_vocabulary()))
        x.append(enc_layers[idx](layer))

    concat_layer = keras.layers.Concatenate()
    x = concat_layer(x)

    cat_emb_layer = CategoricalFeatureEmbedding(
        cardinalities=cardinalities,
        emb_dim=emb_dim
    )
    x = cat_emb_layer(x)

    flatten_layer = keras.layers.Flatten()

    outputs = flatten_layer(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    return model


def build_full_model(
    inter_model: keras.Model,
    users_model: keras.Model = None,
    items_model: keras.Model = None,
    mlp_dims: List[int] = [],
    dropout=0.3
):

    inputs = []
    inputs += inter_model.inputs
    models_outputs = [inter_model.outputs[0]]

    if users_model is not None:
        inputs += users_model.inputs
        models_outputs.append(users_model.outputs[0])

    if items_model is not None:
        inputs += items_model.inputs
        models_outputs.append(items_model.outputs[0])

    x = keras.layers.Concatenate()(models_outputs)

    dims = [x.shape[1], *mlp_dims, 1]

    mlp = MLP(dims, dropout=dropout)

    outputs = mlp(x)

    return keras.Model(inputs=inputs, outputs=outputs)


def split_data(X: Dict[str, np.ndarray], y: np.ndarray, test_size=0.1):
    X_train = {}
    X_test = {}

    split = np.random.choice(range(y.shape[0]), int((1-test_size)*y.shape[0]))

    for k in X:
        X_train[k] = X[k][split]
        X_test[k] = X[k][~split]

    y_train, y_test = y[split], y[~split]

    return X_train, y_train, X_test, y_test
