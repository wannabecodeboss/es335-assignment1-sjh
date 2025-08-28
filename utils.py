"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""
import pandas as pd
import numpy as np


def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """
    X_encoded = pd.DataFrame()
    
    for col in X.columns:
        if not check_ifreal(X[col]):
            # Discrete column - perform one-hot encoding
            one_hot = pd.get_dummies(X[col], prefix=col)
            X_encoded = pd.concat([X_encoded, one_hot], axis=1)
        else:
            # Real-valued column - keep as is
            X_encoded[col] = X[col]
    
    return X_encoded

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    # Check if all values are numeric and have decimal parts or if dtype is float
    if pd.api.types.is_numeric_dtype(y):
        # Check if any value has decimal part or if it's float dtype
        return y.dtype in ['float64', 'float32'] or any(y != y.astype(int))
    return False

def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    if len(Y) == 0:
        return 0
    
    # Get value counts and calculate probabilities
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate entropy: -sum(p * log2(p))
    entropy_val = 0
    for prob in probabilities:
        if prob > 0:  # Avoid log(0)
            entropy_val -= prob * np.log2(prob)
    
    return entropy_val

def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    if len(Y) == 0:
        return 0
    
    # Get value counts and calculate probabilities
    value_counts = Y.value_counts()
    probabilities = value_counts / len(Y)
    
    # Calculate Gini index: 1 - sum(p^2)
    gini = 1 - sum(prob**2 for prob in probabilities)
    
    return gini

def mse(Y: pd.Series) -> float:
    """
    Function to calculate the mean squared error (variance)
    """
    if len(Y) == 0:
        return 0
    
    mean_val = Y.mean()
    mse_val = ((Y - mean_val) ** 2).mean()
    
    return mse_val

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    if len(Y) == 0:
        return 0
    
    # Calculate initial impurity
    if criterion.lower() == 'entropy':
        initial_impurity = entropy(Y)
    elif criterion.lower() == 'gini':
        initial_impurity = gini_index(Y)
    elif criterion.lower() == 'mse':
        initial_impurity = mse(Y)
    else:
        raise ValueError("Criterion must be 'entropy', 'gini', or 'mse'")
    
    # Calculate weighted average impurity after split
    unique_values = attr.unique()
    weighted_impurity = 0
    
    for value in unique_values:
        # Create mask for this value
        mask = (attr == value)
        subset_y = Y[mask]
        
        if len(subset_y) > 0:
            # Calculate impurity for this subset
            if criterion.lower() == 'entropy':
                subset_impurity = entropy(subset_y)
            elif criterion.lower() == 'gini':
                subset_impurity = gini_index(subset_y)
            elif criterion.lower() == 'mse':
                subset_impurity = mse(subset_y)
            
            # Weight by proportion of samples
            weight = len(subset_y) / len(Y)
            weighted_impurity += weight * subset_impurity
    
    # Information gain = initial impurity - weighted average impurity
    info_gain = initial_impurity - weighted_impurity
    
    return info_gain

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.
    features: pd.Series is a list of all the attributes we have to split upon
    return: attribute to split upon and the best split value (for real features)
    """
    best_gain = -float('inf')
    best_attribute = None
    best_split_value = None
    
    for feature in features:
        if feature not in X.columns:
            continue
            
        attr_values = X[feature]
        
        if check_ifreal(attr_values):
            # Real-valued feature: try different split points
            unique_values = sorted(attr_values.unique())
            
            for i in range(len(unique_values) - 1):
                # Try split point between consecutive unique values
                split_point = (unique_values[i] + unique_values[i + 1]) / 2
                
                # Create binary attribute: True if <= split_point, False otherwise
                binary_attr = attr_values <= split_point
                
                # Calculate information gain
                gain = information_gain(y, binary_attr, criterion)
                
                if gain > best_gain:
                    best_gain = gain
                    best_attribute = feature
                    best_split_value = split_point
        else:
            # Discrete feature: use all values directly
            gain = information_gain(y, attr_values, criterion)
            
            if gain > best_gain:
                best_gain = gain
                best_attribute = feature
                best_split_value = None  # No split value for discrete features
    
    return best_attribute, best_split_value

def split_data(X: pd.DataFrame, y: pd.Series, attribute, value):
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.
    attribute: attribute/feature to split upon
    value: value of that attribute to split upon (None for discrete features)
    return: splitted data(Input and output)
    """
    if attribute not in X.columns:
        raise ValueError(f"Attribute {attribute} not found in data")
    
    attr_values = X[attribute]
    
    if value is None:
        # Discrete feature: split by unique values
        splits = {}
        unique_values = attr_values.unique()
        
        for val in unique_values:
            mask = (attr_values == val)
            splits[val] = (X[mask].reset_index(drop=True), y[mask].reset_index(drop=True))
        
        return splits
    else:
        # Real feature: binary split at threshold
        left_mask = attr_values <= value
        right_mask = ~left_mask
        
        left_split = (X[left_mask].reset_index(drop=True), y[left_mask].reset_index(drop=True))
        right_split = (X[right_mask].reset_index(drop=True), y[right_mask].reset_index(drop=True))
        
        return {f'<= {value}': left_split, f'> {value}': right_split}

def calculate_variance_reduction(Y: pd.Series, attr: pd.Series) -> float:
    """
    Helper function to calculate variance reduction for regression trees
    """
    if len(Y) == 0:
        return 0
    
    # Initial variance
    initial_variance = Y.var() if len(Y) > 1 else 0
    
    # Calculate weighted variance after split
    unique_values = attr.unique()
    weighted_variance = 0
    
    for value in unique_values:
        mask = (attr == value)
        subset_y = Y[mask]
        
        if len(subset_y) > 0:
            subset_variance = subset_y.var() if len(subset_y) > 1 else 0
            weight = len(subset_y) / len(Y)
            weighted_variance += weight * subset_variance
    
    # Variance reduction
    variance_reduction = initial_variance - weighted_variance
    
    return variance_reduction