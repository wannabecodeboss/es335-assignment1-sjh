"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
from dataclasses import dataclass
from typing import Literal, Union, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import *
try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False
    print("Warning: graphviz not available. Install with 'pip install graphviz' for better tree visualization.")

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
        # Store feature names for later use
        self.feature_names = list(X.columns)
        
        # Determine if this is a regression or classification problem
        self.is_regression = check_ifreal(y)
        
        # Convert discrete features to one-hot encoding
        X_encoded = one_hot_encoding(X)
        
        # Build the tree
        self.root = self._build_tree(X_encoded, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        """
        Recursive function to build the decision tree
        """
        node = TreeNode()
        node.depth = depth
        
        # Base cases for stopping recursion
        # 1. Maximum depth reached
        if depth >= self.max_depth:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node
        
        # 2. All samples have the same target value
        if len(y.unique()) == 1:
            node.is_leaf = True
            node.prediction = y.iloc[0]
            return node
        
        # 3. No more features to split on
        if X.empty or len(X.columns) == 0:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node
        
        # 4. Minimum samples check
        if len(y) < 2:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node
        
        # Find the best feature to split on
        features = pd.Series(X.columns)
        
        # Determine criterion based on problem type
        if self.is_regression:
            criterion = "mse"
        else:
            criterion = "entropy" if self.criterion == "information_gain" else "gini"
        
        best_feature, best_threshold = opt_split_attribute(X, y, criterion, features)
        
        # If no good split found, make leaf
        if best_feature is None:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
            return node
        
        # Set node properties
        node.feature = best_feature
        node.threshold = best_threshold
        
        # Split the data
        splits = split_data(X, y, best_feature, best_threshold)
        
        # Create children nodes
        for split_value, (X_subset, y_subset) in splits.items():
            if len(y_subset) > 0:  # Only create child if subset is not empty
                child = self._build_tree(X_subset, y_subset, depth + 1)
                node.children[split_value] = child
        
        # If no children were created, make this a leaf
        if not node.children:
            node.is_leaf = True
            node.prediction = self._make_prediction(y)
        
        return node

    def _make_prediction(self, y: pd.Series) -> Union[float, str, int]:
        """
        Make prediction for a leaf node
        """
        if self.is_regression:
            return y.mean()
        else:
            return y.mode().iloc[0] if not y.empty else 0

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs
        """
        if self.root is None:
            raise ValueError("Tree has not been fitted yet. Call fit() first.")
        
        # Convert discrete features to one-hot encoding (same as training)
        X_encoded = one_hot_encoding(X)
        
        # Make predictions for each sample
        predictions = []
        for idx in range(len(X_encoded)):
            sample = X_encoded.iloc[idx]
            prediction = self._predict_sample(sample, self.root)
            predictions.append(prediction)
        
        return pd.Series(predictions)

    def _predict_sample(self, sample: pd.Series, node: TreeNode) -> Union[float, str, int]:
        """
        Predict a single sample by traversing the tree
        """
        if node.is_leaf:
            return node.prediction
        
        feature_value = sample.get(node.feature, 0)  # Default to 0 if feature not found
        
        if node.threshold is not None:
            # Real-valued feature with binary split
            if feature_value <= node.threshold:
                split_key = f'<= {node.threshold}'
            else:
                split_key = f'> {node.threshold}'
        else:
            # Discrete feature - find exact match
            split_key = None
            for key in node.children.keys():
                if str(feature_value) == str(key) or feature_value == key:
                    split_key = key
                    break
            
            # If no exact match found, use the most common class/value
            if split_key is None:
                return node.prediction if hasattr(node, 'prediction') else self._get_default_prediction()
        
        # Traverse to child node
        if split_key in node.children:
            return self._predict_sample(sample, node.children[split_key])
        else:
            # If split key not found, return default prediction
            return node.prediction if hasattr(node, 'prediction') else self._get_default_prediction()

    def _get_default_prediction(self):
        """
        Get default prediction when no path is found
        """
        if self.is_regression:
            return 0.0
        else:
            return "Unknown"

    def plot(self, method="graphviz", filename="decision_tree", view=True) -> None:
        """
        Function to plot the tree
        
        Parameters:
        method: str, either "graphviz" (default) or "text"
        filename: str, filename for graphviz output (without extension)
        view: bool, whether to automatically open the generated file
        
        Output Example (text mode):
        ?(X1 > 4)
            Y: ?(X2 > 7)
                Y: Class A
                N: Class B
            N: Class C
        Where Y => Yes and N => No
        """
        if self.root is None:
            print("Tree has not been fitted yet. Call fit() first.")
            return
        
        if method == "graphviz" and GRAPHVIZ_AVAILABLE:
            self._plot_graphviz(filename, view)
        else:
            if method == "graphviz" and not GRAPHVIZ_AVAILABLE:
                print("Graphviz not available, falling back to text visualization.")
            print("Decision Tree Structure:")
            print("=" * 50)
            self._plot_node(self.root, "", True)

    def _plot_node(self, node: TreeNode, prefix: str, is_root: bool = False) -> None:
        """
        Recursively plot the tree structure
        """
        if node.is_leaf:
            print(f"{prefix}Prediction: {node.prediction}")
            return
        
        # Print the split condition
        if node.threshold is not None:
            condition = f"{node.feature} > {node.threshold:.3f}"
        else:
            condition = f"{node.feature}"
        
        if is_root:
            print(f"?({condition})")
        else:
            print(f"{prefix}?({condition})")
        
        # Plot children
        children_items = list(node.children.items())
        for i, (split_value, child) in enumerate(children_items):
            is_last = (i == len(children_items) - 1)
            
            if node.threshold is not None:
                # Binary split for real features
                branch_label = "Y" if ">" in str(split_value) else "N"
            else:
                # Multi-way split for discrete features
                branch_label = f"{split_value}"
            
            # Create prefix for child
            if is_last:
                child_prefix = prefix + "    "
                connector = "└── "
            else:
                child_prefix = prefix + "│   "
                connector = "├── "
            
            print(f"{prefix}{connector}{branch_label}:", end=" ")
            
            if child.is_leaf:
                print(f"{child.prediction}")
            else:
                print()
                self._plot_node(child, child_prefix, False)

    def get_tree_depth(self) -> int:
        """
        Get the actual depth of the tree
        """
        if self.root is None:
            return 0
        return self._get_node_depth(self.root)

    def _get_node_depth(self, node: TreeNode) -> int:
        """
        Recursively calculate the depth of a node
        """
        if node.is_leaf:
            return 1
        
        max_child_depth = 0
        for child in node.children.values():
            child_depth = self._get_node_depth(child)
            max_child_depth = max(max_child_depth, child_depth)
        
        return 1 + max_child_depth

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Calculate feature importance based on how often features are used for splitting
        """
        if self.root is None:
            return {}
        
        importance = {}
        self._calculate_importance(self.root, importance)
        
        # Normalize importance scores
        total_importance = sum(importance.values())
        if total_importance > 0:
            for feature in importance:
                importance[feature] /= total_importance
        
        return importance

    def _calculate_importance(self, node: TreeNode, importance: Dict[str, float]) -> None:
        """
        Recursively calculate feature importance
        """
        if not node.is_leaf and node.feature:
            importance[node.feature] = importance.get(node.feature, 0) + 1
            
            for child in node.children.values():
                self._calculate_importance(child, importance)

    def _plot_graphviz(self, filename="decision_tree", view=True):
        """
        Create a Graphviz visualization of the decision tree with metrics
        """
        dot = graphviz.Digraph(comment='Decision Tree')
        dot.attr(rankdir='TB')  # Top to Bottom layout
        dot.attr('node', shape='box', style='rounded,filled', fontname='Arial', fontsize='10')
        dot.attr('edge', fontname='Arial', fontsize='9')
        
        # Calculate tree metrics
        tree_depth = self.get_tree_depth()
        total_nodes = self._count_nodes(self.root)
        leaf_nodes = self._count_leaf_nodes(self.root)
        feature_importance = self.get_feature_importance()
        
        # Create title with metrics
        title = f"Decision Tree\\n"
        title += f"Max Depth: {self.max_depth} | Actual Depth: {tree_depth}\\n"
        title += f"Criterion: {self.criterion} | Nodes: {total_nodes} | Leaves: {leaf_nodes}\\n"
        title += f"Problem Type: {'Regression' if self.is_regression else 'Classification'}\\n"
        
        # Add feature importance if available
        if feature_importance:
            top_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:3]
            importance_str = " | ".join([f"{feat}: {imp:.3f}" for feat, imp in top_features])
            title += f"Top Features: {importance_str}"
        
        # Add title as a special node
        dot.node('title', title, shape='plaintext', fontsize='12', fontname='Arial Bold')
        
        # Add nodes and edges
        root_id = self._add_graphviz_nodes(dot, self.root, node_id=1)
        
        # Connect title to root
        dot.edge('title', '1', style='invisible')
        
        # Set graph attributes for better layout
        dot.attr(size='12,8!')
        dot.attr(dpi='300')
        
        # Render the tree
        try:
            dot.render(filename, format='png', cleanup=True, view=view)
            print(f"Tree visualization with metrics saved as {filename}.png")
            if view:
                print(f"Opening {filename}.png...")
        except Exception as e:
            print(f"Error rendering tree: {e}")
            print("Falling back to text visualization:")
            self.plot(method="text")
    
    def _add_graphviz_nodes(self, dot, node, node_id=1, parent_id=None, edge_label=""):
        """
        Recursively add nodes and edges to the Graphviz diagram with enhanced information
        """
        current_id = node_id
        
        if node.is_leaf:
            # Leaf node - show prediction with additional info
            if self.is_regression:
                label = f"Prediction: {node.prediction:.3f}\\nLeaf Node\\nDepth: {node.depth}"
                color = 'lightblue'
            else:
                label = f"Class: {node.prediction}\\nLeaf Node\\nDepth: {node.depth}"
                color = 'lightgreen'
            
            dot.node(str(current_id), label, fillcolor=color, fontsize='9')
        else:
            # Internal node - show split condition with additional info
            if node.threshold is not None:
                condition = f"{node.feature} <= {node.threshold:.3f}"
                split_type = "Binary Split"
            else:
                condition = f"{node.feature}"
                split_type = f"{len(node.children)}-way Split"
            
            label = f"{condition}\\n{split_type}\\nDepth: {node.depth}"
            
            dot.node(str(current_id), label, fillcolor='lightcoral', fontsize='9')
        
        # Add edge from parent if exists
        if parent_id is not None:
            dot.edge(str(parent_id), str(current_id), label=edge_label, fontsize='8')
        
        # Add children
        if not node.is_leaf:
            child_id = current_id + 1
            for split_value, child in node.children.items():
                # Determine edge label with enhanced information
                if hasattr(node, 'threshold') and node.threshold is not None:
                    if ">" in str(split_value):
                        edge_label = f"False\\n(> {node.threshold:.3f})"
                    else:
                        edge_label = f"True\\n(<= {node.threshold:.3f})"
                else:
                    edge_label = f"= {split_value}"
                
                child_id = self._add_graphviz_nodes(dot, child, child_id, current_id, edge_label)
        
        return current_id + 1
    
    def _count_nodes(self, node):
        """
        Count total nodes in subtree
        """
        if node.is_leaf:
            return 1
        
        count = 1
        for child in node.children.values():
            count += self._count_nodes(child)
        
        return count
    
    def _count_leaf_nodes(self, node):
        """
        Count total leaf nodes in subtree
        """
        if node.is_leaf:
            return 1
        
        count = 0
        for child in node.children.values():
            count += self._count_leaf_nodes(child)
        
        return count