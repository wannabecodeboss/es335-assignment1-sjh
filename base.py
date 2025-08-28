"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *

np.random.seed(42)

class TreeNode:
    """
    Class to represent a node in the decision tree
    """
    def __init__(self):
        self.is_leaf = False
        self.feature = None
        self.threshold = None  # For real-valued features
        self.children = {}  # Dictionary to store children nodes
        self.prediction = None  # For leaf nodes
        self.depth = 0

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index"]  # criterion won't be used for regression
    max_depth: int  # The maximum depth the tree can grow to

    def __init__(self, criterion="information_gain", max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.feature_names = None
        self.is_regression = False

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree
        """
        self.feature_names = list(X.columns)
        self.is_regression = check_ifreal(y)
        X_encoded = one_hot_encoding(X)
        self.root = self._build_tree(X_encoded, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        node = TreeNode()
        node.depth = depth

        # Base cases
        if depth >= self.max_depth or len(y.unique()) == 1 or X.empty or len(X.columns) == 0 or len(y) < 2:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node

        # Choose best split
        features = pd.Series(X.columns)
        criterion = "mse" if self.is_regression else ("entropy" if self.criterion == "information_gain" else "gini")
        best_feature, best_threshold = opt_split_attribute(X, y, criterion, features)

        if best_feature is None:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node

        node.feature = best_feature
        node.threshold = best_threshold
        splits = split_data(X, y, best_feature, best_threshold)

        for split_value, (X_subset, y_subset) in splits.items():
            if len(y_subset) > 0:
                child = self._build_tree(X_subset, y_subset, depth + 1)
                node.children[split_value] = child

        if not node.children:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)

        return node

    def _make_prediction(self, y: pd.Series) -> Union[float, str, int]:
        if self.is_regression:
            return y.mean()
        else:
            return y.mode().iloc[0] if not y.empty else 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        X_encoded = one_hot_encoding(X)
        predictions = []
        for idx in range(len(X_encoded)):
            sample = X_encoded.iloc[idx]
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        return pd.Series(predictions)

    def _predict_sample(self, sample: pd.Series, node: TreeNode) -> Union[float, str, int]:
        if node.is_leaf:
            return node.prediction

        feature_value = sample.get(node.feature, 0)
        if node.threshold is not None:
            if feature_value <= node.threshold:
                split_key = f'<= {node.threshold}'
            else:
                split_key = f'> {node.threshold}'
        else:
            split_key = None
            for key in node.children.keys():
                if str(feature_value) == str(key) or feature_value == key:
                    split_key = key
                    break
            if split_key is None:
                return node.prediction if hasattr(node, 'prediction') else self._get_default_prediction()

        if split_key in node.children:
            return self._predict_sample(sample, node.children[split_key])
        else:
            return node.prediction if hasattr(node, 'prediction') else self._get_default_prediction()

    def _get_default_prediction(self):
        return 0.0 if self.is_regression else "Unknown"

    def plot(self) -> None:
        if self.root is None:
            print("Tree has not been fitted yet. Call fit() first.")
            return
        print("Decision Tree Structure:")
        print("=" * 50)
        self._plot_node(self.root, "", True)

    def _plot_node(self, node: TreeNode, prefix: str, is_root: bool = False, condition_label: str = "") -> None:
        if node.is_leaf:
            if condition_label:
                print(f"{prefix}{condition_label} -> Prediction: {node.prediction}")
            else:
                print(f"{prefix}Prediction: {node.prediction}")
            return

        if node.threshold is not None:
            condition = f"{node.feature} > {node.threshold:.3f}"
        else:
            condition = f"{node.feature}"

        if is_root:
            print(f"?({condition})")
        else:
            print(f"{prefix}?({condition})")

        children_items = list(node.children.items())
        for i, (split_value, child) in enumerate(children_items):
            is_last = (i == len(children_items) - 1)

            if node.threshold is not None:
                branch_label = "Y" if ">" in str(split_value) else "N"
            else:
                branch_label = f"{split_value}"

            if is_last:
                child_prefix = prefix + "    "
                connector = "└── "
            else:
                child_prefix = prefix + "│   "
                connector = "├── "

            if child.is_leaf:
                print(f"{prefix}{connector}{branch_label} -> Prediction: {child.prediction}")
            else:
                print(f"{prefix}{connector}{branch_label}:")
                self._plot_node(child, child_prefix, False)

    def get_tree_depth(self) -> int:
        if self.root is None:
            return 0
        return self._get_node_depth(self.root)

    def _get_node_depth(self, node: TreeNode) -> int:
        if node.is_leaf:
            return 1
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_node_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        return 1 + max_child_depth
