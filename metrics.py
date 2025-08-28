from typing import Union
import pandas as pd
import numpy as np

def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """
    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    
    # Calculate accuracy as the fraction of correct predictions
    correct_predictions = (y_hat == y).sum()
    total_predictions = len(y)
    
    return correct_predictions / total_predictions

def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    assert cls in y_hat.values or cls in y.values, f"Class {cls} not found in predictions or actual values"
    
    # True Positives: correctly predicted as positive class
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    
    # False Positives: incorrectly predicted as positive class
    false_positives = ((y_hat == cls) & (y != cls)).sum()
    
    # Predicted Positives: all instances predicted as positive class
    predicted_positives = true_positives + false_positives
    
    # Handle edge case when no predictions for this class
    if predicted_positives == 0:
        return 0.0
    
    precision_score = true_positives / predicted_positives
    return precision_score

def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    assert cls in y_hat.values or cls in y.values, f"Class {cls} not found in predictions or actual values"
    
    # True Positives: correctly predicted as positive class
    true_positives = ((y_hat == cls) & (y == cls)).sum()
    
    # False Negatives: incorrectly predicted as negative class
    false_negatives = ((y_hat != cls) & (y == cls)).sum()
    
    # Actual Positives: all instances that are actually positive class
    actual_positives = true_positives + false_negatives
    
    # Handle edge case when no actual instances of this class
    if actual_positives == 0:
        return 0.0
    
    recall_score = true_positives / actual_positives
    return recall_score

def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    
    # Check if values are numeric
    assert pd.api.types.is_numeric_dtype(y_hat), "Predictions must be numeric for RMSE calculation"
    assert pd.api.types.is_numeric_dtype(y), "Actual values must be numeric for RMSE calculation"
    
    # Calculate squared differences
    squared_errors = (y_hat - y) ** 2
    
    # Calculate mean squared error
    mse = squared_errors.mean()
    
    # Calculate root mean squared error
    rmse_score = np.sqrt(mse)
    
    return rmse_score

def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    
    # Check if values are numeric
    assert pd.api.types.is_numeric_dtype(y_hat), "Predictions must be numeric for MAE calculation"
    assert pd.api.types.is_numeric_dtype(y), "Actual values must be numeric for MAE calculation"
    
    # Calculate absolute differences
    absolute_errors = np.abs(y_hat - y)
    
    # Calculate mean absolute error
    mae_score = absolute_errors.mean()
    
    return mae_score

def f1_score(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the F1 score (harmonic mean of precision and recall)
    """
    precision_score = precision(y_hat, y, cls)
    recall_score = recall(y_hat, y, cls)
    
    # Handle edge case when both precision and recall are 0
    if precision_score + recall_score == 0:
        return 0.0
    
    f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score)
    return f1

def confusion_matrix(y_hat: pd.Series, y: pd.Series) -> pd.DataFrame:
    """
    Function to calculate the confusion matrix
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    
    # Get unique classes from both actual and predicted values
    classes = sorted(list(set(y.unique()) | set(y_hat.unique())))
    
    # Initialize confusion matrix
    cm = pd.DataFrame(0, index=classes, columns=classes)
    cm.index.name = 'Actual'
    cm.columns.name = 'Predicted'
    
    # Fill confusion matrix
    for actual, predicted in zip(y, y_hat):
        cm.loc[actual, predicted] += 1
    
    return cm

def classification_report(y_hat: pd.Series, y: pd.Series) -> pd.DataFrame:
    """
    Function to generate a comprehensive classification report
    """
    assert y_hat.size == y.size, "Size of predictions and actual values must be equal"
    assert y_hat.size > 0, "Input series cannot be empty"
    
    # Get unique classes
    classes = sorted(list(set(y.unique()) | set(y_hat.unique())))
    
    # Calculate metrics for each class
    report_data = []
    for cls in classes:
        prec = precision(y_hat, y, cls)
        rec = recall(y_hat, y, cls)
        f1 = f1_score(y_hat, y, cls)
        support = (y == cls).sum()
        
        report_data.append({
            'Class': cls,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'Support': support
        })
    
    # Add overall metrics
    overall_accuracy = accuracy(y_hat, y)
    macro_avg_precision = np.mean([row['Precision'] for row in report_data])
    macro_avg_recall = np.mean([row['Recall'] for row in report_data])
    macro_avg_f1 = np.mean([row['F1-Score'] for row in report_data])
    
    report_data.append({
        'Class': 'Macro Avg',
        'Precision': macro_avg_precision,
        'Recall': macro_avg_recall,
        'F1-Score': macro_avg_f1,
        'Support': len(y)
    })
    
    report_data.append({
        'Class': 'Accuracy',
        'Precision': overall_accuracy,
        'Recall': overall_accuracy,
        'F1-Score': overall_accuracy,
        'Support': len(y)
    })
    
    return pd.DataFrame(report_data)