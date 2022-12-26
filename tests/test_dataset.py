from deeprec.dataset import (
    RecDataset
)
import pytest
import pandas as pd
import numpy as np


class TestRecDataset:

    @classmethod
    def setup_class(cls):
        """ setup any state specific to the execution of the given class (which
        usually contains tests).
        """
        pass
    
    @classmethod
    def teardown_class(cls):
        """ teardown any state that was previously setup with a call to
        setup_class.
        """
        pass

    def test_dataset_init(self):
        ds = RecDataset();
        assert isinstance(ds, RecDataset)


    def test_dataset_init_with_params(self):
        ds = RecDataset(
            user_id='user_idx',
            item_id='item_idx',
            target_id='rating'
        );
        assert isinstance(ds, RecDataset)
        assert ds.user_id == 'user_idx'
        assert ds.item_id == 'item_idx'
        assert ds.target_id == 'rating'
        
        
    def test_dataset_set_params(self):
        ds = RecDataset()
        
        params = {
            "user_id":'user_idx',
            "item_id":'item_idx',
            "target_id":'rating'
        }
        
        ds.set_params(**params)
        
        assert isinstance(ds, RecDataset)
        assert ds.user_id == 'user_idx'
        assert ds.item_id == 'item_idx'
        assert ds.target_id == 'rating'
        

        
    def test_dataset_get_params(self):
        ds = RecDataset(
            user_id='user_idx',
            item_id='item_idx',
            target_id='rating'
        );
        
        params = ds.get_params()
        
        assert params.get("user_id") == 'user_idx'
        assert params.get("item_id") == 'item_idx'
        assert params.get("target_id")  == 'rating'
        
        
    def test_dataset_call(self):
        inter = pd.DataFrame({
            "user_idx":[1,1,1,1,2,2,2,2],
            "item_idx":[1,2,3,4,1,2,3,4],
            "rating": [5,4,3,2,2,3,4,5]
        })
        ds = RecDataset(
            user_id='user_idx',
            item_id='item_idx',
            target_id='rating'
        );
        X, y = ds(inter)

        assert isinstance(X, dict)
        assert isinstance(y, np.ndarray)
        
        
    def test_dataset_call_with_features(self):
        inter = pd.DataFrame({
            "user_idx":[1,1,1,1,2,2,2,2],
            "item_idx":[1,2,3,4,1,2,3,4],
            "rating": [5,4,3,2,2,3,4,5]
        })
        
        users_feat = pd.DataFrame({
            "user_idx":[1,2],
            "u_feature":["cat1","cat2"]
        })
        
        items_feat = pd.DataFrame({
            "item_idx":[1,2,3,4],
            "i_feature": ["cat1","cat2","cat1","cat2"]
        })
        
        ds = RecDataset(
            user_id='user_idx',
            item_id='item_idx',
            target_id='rating'
        );
        
        X, y = ds(inter, users_feat, items_feat)

        assert isinstance(X, dict)
        assert isinstance(y, np.ndarray)