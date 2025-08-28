import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
data = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

print("Original dataset shape:", data.shape)
print("Original columns:", data.columns.tolist())
print("\nFirst few rows:")
print(data.head())

# Clean the above data by removing redundant columns and rows with junk values
print("\n" + "="*50)
print("DATA CLEANING")
print("="*50)

# Remove rows with missing horsepower (marked as '?')
print("Rows with '?' in horsepower:", (data['horsepower'] == '?').sum())
data_clean = data[data['horsepower'] != '?'].copy()
data_clean['horsepower'] = pd.to_numeric(data_clean['horsepower'])

# Remove redundant columns - 'car name' is not useful for prediction
data_clean = data_clean.drop('car name', axis=1)

print("Cleaned dataset shape:", data_clean.shape)
print("Cleaned columns:", data_clean.columns.tolist())
print("No missing values:", data_clean.isnull().sum().sum() == 0)

# =============================================================================
# Q1a) Show usage of decision tree for automotive efficiency problem
# =============================================================================

print("\n" + "="*60)
print("Q1a) USAGE OF OUR DECISION TREE FOR AUTOMOTIVE EFFICIENCY")
print("="*60)

# Prepare data
X = data_clean.drop('mpg', axis=1)  # Features
y = data_clean['mpg']               # Target (continuous - regression)

print("Features:", X.columns.tolist())
print("Target: mpg (regression problem)")
print("Dataset size:", len(X))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Train our decision tree
print("\nTraining our Decision Tree...")
our_tree = DecisionTree(criterion="information_gain", max_depth=5)
our_tree.fit(X_train, y_train)

# Make predictions
our_predictions = our_tree.predict(X_test)

# Calculate performance metrics
our_rmse = rmse(our_predictions, y_test)
our_mae = mae(our_predictions, y_test)

print(f"\nOur Decision Tree Results:")
print(f"RMSE: {our_rmse:.3f}")
print(f"MAE:  {our_mae:.3f}")

# Show some predictions
print(f"\nSample Predictions:")
for i in range(5):
    pred = our_predictions.iloc[i] if hasattr(our_predictions, 'iloc') else our_predictions[i]
    actual = y_test.iloc[i]
    print(f"Predicted: {pred:.2f}, Actual: {actual:.2f}")

# Visualize results
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_test, our_predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Our Decision Tree\nRMSE: {our_rmse:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.hist(y_test - our_predictions, bins=15, alpha=0.7)
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.title('Prediction Errors')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================================================
# Q1b) Compare performance with scikit-learn decision tree
# =============================================================================

print("\n" + "="*60)
print("Q1b) COMPARISON WITH SCIKIT-LEARN DECISION TREE")
print("="*60)

# Train scikit-learn decision tree with same parameters
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X_train, y_train)

# Make predictions
sklearn_predictions = sklearn_tree.predict(X_test)

# Calculate performance metrics
sklearn_rmse = rmse(pd.Series(sklearn_predictions), y_test)
sklearn_mae = mae(pd.Series(sklearn_predictions), y_test)

print(f"scikit-learn Decision Tree Results:")
print(f"RMSE: {sklearn_rmse:.3f}")
print(f"MAE:  {sklearn_mae:.3f}")

# Performance comparison
print(f"\n" + "="*40)
print("PERFORMANCE COMPARISON")
print("="*40)
print(f"{'Metric':<10} {'Our Tree':<12} {'sklearn':<12} {'Difference':<12}")
print("-" * 50)
print(f"{'RMSE':<10} {our_rmse:<12.3f} {sklearn_rmse:<12.3f} {our_rmse-sklearn_rmse:<12.3f}")
print(f"{'MAE':<10} {our_mae:<12.3f} {sklearn_mae:<12.3f} {our_mae-sklearn_mae:<12.3f}")

# Visual comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, our_predictions, alpha=0.7, label='Our Tree')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'Our Decision Tree\nRMSE: {our_rmse:.3f}')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(y_test, sklearn_predictions, alpha=0.7, label='sklearn', color='orange')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Actual MPG')
plt.ylabel('Predicted MPG')
plt.title(f'scikit-learn Decision Tree\nRMSE: {sklearn_rmse:.3f}')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Summary
print(f"\nSUMMARY:")
print(f"Dataset: {len(data_clean)} cars, {X.shape[1]} features")
print(f"Problem: Predicting MPG (regression)")
if our_rmse < sklearn_rmse:
    print("Our implementation performs BETTER than scikit-learn")
elif our_rmse > sklearn_rmse:
    print("scikit-learn performs BETTER than our implementation")
else:
    print("Both implementations perform similarly")

print(f"Difference in RMSE: {abs(our_rmse - sklearn_rmse):.3f}")
