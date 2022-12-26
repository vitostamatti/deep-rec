from deeprec.model import RecModel
from deeprec.dataset import RecDataset
from tensorflow import keras
import pandas as pd

import pytest


@pytest.fixture
def users_data():
    return pd.Series([1, 2], name='user_idx')


@pytest.fixture
def items_data():
    return pd.Series([1, 2, 3, 4], name='item_idx')


@pytest.fixture
def features_data():
    users_feat = pd.DataFrame(
        {"user_feature": ["cat1", "cat2"]},
        index=pd.Index([1, 2], name='user_idx')
    )

    items_feat = pd.DataFrame(
        {"item_feature": [
            "cat1", "cat2", "cat1", "cat2"]},
        index=pd.Index([1, 2, 3, 4], name='item_idx')
    )
    return users_feat, items_feat


@ pytest.fixture
def inter_data():
    inter = pd.DataFrame(
        {
            "user_idx": [1, 1, 1, 1, 2, 2, 2, 2],
            "item_idx": [1, 2, 3, 4, 1, 2, 3, 4],
            "rating": [5, 4, 3, 2, 2, 3, 4, 5],
        }
    )
    return inter


@ pytest.fixture
def rec_data(inter_data, features_data):
    users_feat, items_feat = features_data
    ds = RecDataset(user_id="user_idx",
                    item_id="item_idx", target_id="rating")
    X, y = ds(inter_data, users_feat, items_feat)
    return X, y


class TestRecModel:

    def test_init(self):
        recmodel = RecModel()
        assert isinstance(recmodel, RecModel)

    def test_build(self, users_data, items_data, features_data):
        recmodel = RecModel()

        users_feat, items_feat = features_data

        recmodel.build(
            users=users_data,
            items=items_data,
            user_features=users_feat,
            item_features=items_feat,
        )
        assert isinstance(recmodel.model, keras.Model)

    def test_fit(self, users_data, items_data, features_data, rec_data):
        recmodel = RecModel()

        users_feat, items_feat = features_data

        recmodel.build(
            users=users_data,
            items=items_data,
            user_features=users_feat,
            item_features=items_feat,
        )

        X, y = rec_data

        res = recmodel.fit(
            X,
            y,
            val_size=0.1,
            learning_rate=1e-4,
            epochs=1,
            batch_size=1,
            verbose=2,
            patience=3
        )
        assert isinstance(res.history, dict)

    def test_predict(self, users_data, items_data, features_data, rec_data, inter_data):
        recmodel = RecModel()

        users_feat, items_feat = features_data

        recmodel.build(
            users=users_data,
            items=items_data,
            user_features=users_feat,
            item_features=items_feat,
        )

        X, y = rec_data

        _ = recmodel.fit(
            X,
            y,
            val_size=0.1,
            learning_rate=1e-4,
            epochs=1,
            batch_size=1,
            verbose=2,
            patience=3
        )
        preds = recmodel.predict(X)

        assert (preds.shape[0] == y.shape[0])
