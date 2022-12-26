from typing import Dict
import pandas as pd
import numpy as np


class RecDataset:

    def __init__(
        self,
        user_id: str = "user_id",
        item_id: str = "item_id",
        target_id: str = "rating",
    ):

        self.user_id = user_id
        self.item_id = item_id
        self.target_id = target_id

    def get_params(self):
        return {
            "user_id": self.user_id,
            "item_id": self.item_id,
            "target_id": self.target_id,
        }

    def set_params(
        self,
        user_id: str = None,
        item_id: str = None,
        target_id: str = None
    ):
        self.user_id = user_id if user_id else self.user_id
        self.item_id = item_id if item_id else self.item_id
        self.target_id = target_id if target_id else self.target_id

    def build_x(self, X: pd.DataFrame):
        x = {}
        for i in X.columns:
            x[i] = X[i].values
        return x

    def build_y(self, y: pd.Series):
        return y.values

    def build(self, X: pd.DataFrame = None, y: pd.Series = None):
        x = self.build_x(X)
        y = self.build_y(y)
        return x, y

    def __call__(
        self,
        interactions: pd.DataFrame,
        user_features: pd.DataFrame = None,
        item_features: pd.DataFrame = None,
    ):

        X = interactions[[self.user_id, self.item_id]].copy()

        y = interactions[self.target_id].copy()

        if isinstance(user_features, pd.DataFrame):
            X = X.merge(
                user_features, left_on=self.user_id, right_on=self.user_id, how="left"
            )

        if isinstance(item_features, pd.DataFrame):
            X = X.merge(
                item_features, left_on=self.item_id, right_on=self.item_id, how="left"
            )

        return self.build(X, y)



