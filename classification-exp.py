import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.figure(figsize=(10, 8))
plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Classification Dataset')
plt.colorbar(label='Class')
plt.grid(True, alpha=0.3)
plt.show()

print("Dataset Info:")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")
print(f"Classes: {np.unique(y)}")
print(f"Class distribution: {np.bincount(y)}")

# =============================================================================
# Q2 a) 70/30 Train-Test Split Evaluation
# =============================================================================

print("\n" + "="*60)
print("Q2 a) 70/30 Train-Test Split Evaluation")
print("="*60)

# Convert to pandas for our decision tree
X_df = pd.DataFrame(X, columns=['Feature_1', 'Feature_2'])
y_series = pd.Series(y)

# Split the data: 70% training, 30% testing
X_train, X_test, y_train, y_test = train_test_split(
    X_df, y_series, test_size=0.3, random_state=42, stratify=y_series
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Training class distribution: {y_train.value_counts().sort_index().values}")
print(f"Testing class distribution: {y_test.value_counts().sort_index().values}")

# Train our decision tree
print("\nTraining Decision Tree...")
dt = DecisionTree(criterion="information_gain", max_depth=5)
dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Calculate metrics
test_accuracy = accuracy(y_pred, y_test)
print(f"\nTest Accuracy: {test_accuracy:.4f}")

# Calculate per-class precision and recall
unique_classes = sorted(y_test.unique())
print("\nPer-class Performance:")
print("-" * 40)
print(f"{'Class':<8} {'Precision':<12} {'Recall':<10}")
print("-" * 40)

for cls in unique_classes:
    prec = precision(y_pred, y_test, cls)
    rec = recall(y_pred, y_test, cls)
    print(f"{cls:<8} {prec:<12.4f} {rec:<10.4f}")

# Generate comprehensive classification report
report = classification_report(y_pred, y_test)
print("\nDetailed Classification Report:")
print(report)

# Plot the decision boundary
def plot_decision_boundary(X, y, model, title="Decision Boundary"):
    plt.figure(figsize=(12, 5))
    
    # Original data plot
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data')
    plt.colorbar(label='Class')
    plt.grid(True, alpha=0.3)
    
    # Decision boundary plot
    plt.subplot(1, 2, 2)
    h = 0.02  # step size
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    mesh_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['Feature_1', 'Feature_2'])
    Z = model.predict(mesh_points)
    Z = pd.Series(Z).astype(float).values.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    plt.scatter(X[:, 0], X[:, 1], c=y, alpha=0.7, edgecolors='black')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'{title} - Test Accuracy: {test_accuracy:.3f}')
    plt.colorbar(label='Predicted Class')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Plot decision boundary
plot_decision_boundary(X_test.values, y_test.values, dt, "Decision Tree Decision Boundary")

# Visualize the tree
print("\nDecision Tree Visualization:")
dt.plot(filename="classification_tree", view=False)

# =============================================================================
# Q2 b) 5-Fold Cross-Validation with Nested CV for Optimal Depth
# =============================================================================

print("\n" + "="*60)
print("Q2 b) 5-Fold Cross-Validation with Nested CV for Optimal Depth")
print("="*60)

def nested_cross_validation(X, y, depth_range, outer_cv=5, inner_cv=5):
    """
    Perform nested cross-validation to find optimal tree depth
    """
    # Outer CV for model evaluation
    outer_kf = KFold(n_splits=outer_cv, shuffle=True, random_state=42)
    
    outer_scores = []
    best_depths = []
    
    fold = 1
    for train_idx, test_idx in outer_kf.split(X):
        print(f"\nOuter Fold {fold}/{outer_cv}")
        print("-" * 30)
        
        # Split data
        X_outer_train, X_outer_test = X.iloc[train_idx], X.iloc[test_idx]
        y_outer_train, y_outer_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner CV for hyperparameter tuning
        inner_kf = KFold(n_splits=inner_cv, shuffle=True, random_state=42)
        
        best_score = -1
        best_depth = None
        depth_scores = {}
        
        for depth in depth_range:
            inner_scores = []
            
            for inner_train_idx, inner_val_idx in inner_kf.split(X_outer_train):
                # Inner split
                X_inner_train = X_outer_train.iloc[inner_train_idx]
                X_inner_val = X_outer_train.iloc[inner_val_idx]
                y_inner_train = y_outer_train.iloc[inner_train_idx]
                y_inner_val = y_outer_train.iloc[inner_val_idx]
                
                # Train model
                dt_inner = DecisionTree(criterion="information_gain", max_depth=depth)
                dt_inner.fit(X_inner_train, y_inner_train)
                
                # Validate
                y_inner_pred = dt_inner.predict(X_inner_val)
                score = accuracy(y_inner_pred, y_inner_val)
                inner_scores.append(score)
            
            avg_score = np.mean(inner_scores)
            depth_scores[depth] = avg_score
            
            if avg_score > best_score:
                best_score = avg_score
                best_depth = depth
        
        print(f"Depth scores: {depth_scores}")
        print(f"Best depth for fold {fold}: {best_depth} (score: {best_score:.4f})")
        
        # Train final model on outer training set with best depth
        dt_final = DecisionTree(criterion="information_gain", max_depth=best_depth)
        dt_final.fit(X_outer_train, y_outer_train)
        
        # Evaluate on outer test set
        y_outer_pred = dt_final.predict(X_outer_test)
        outer_score = accuracy(y_outer_pred, y_outer_test)
        
        outer_scores.append(outer_score)
        best_depths.append(best_depth)
        
        print(f"Outer test score: {outer_score:.4f}")
        
        fold += 1
    
    return outer_scores, best_depths

# Define depth range to search
depth_range = range(1, 11)  # Test depths 1 to 10

print("Performing Nested Cross-Validation...")
print(f"Outer CV folds: 5")
print(f"Inner CV folds: 5") 
print(f"Depth range: {list(depth_range)}")

outer_scores, best_depths = nested_cross_validation(X_df, y_series, depth_range)

# Results analysis
print("\n" + "="*50)
print("NESTED CROSS-VALIDATION RESULTS")
print("="*50)

print(f"Outer CV Scores: {[f'{score:.4f}' for score in outer_scores]}")
print(f"Best Depths per Fold: {best_depths}")
print(f"Mean CV Score: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores)*2:.4f})")
print(f"Most Common Optimal Depth: {max(set(best_depths), key=best_depths.count)}")

# Additional analysis: Regular 5-fold CV for different depths
print("\n" + "="*50)
print("REGULAR 5-FOLD CV ANALYSIS (for comparison)")
print("="*50)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
depth_cv_scores = {}

for depth in depth_range:
    scores = []
    for train_idx, val_idx in kf.split(X_df):
        X_train_cv, X_val_cv = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train_cv, y_val_cv = y_series.iloc[train_idx], y_series.iloc[val_idx]
        
        dt_cv = DecisionTree(criterion="information_gain", max_depth=depth)
        dt_cv.fit(X_train_cv, y_train_cv)
        
        y_pred_cv = dt_cv.predict(X_val_cv)
        score = accuracy(y_pred_cv, y_val_cv)
        scores.append(score)
    
    depth_cv_scores[depth] = {
        'mean': np.mean(scores),
        'std': np.std(scores)
    }

# Plot CV results
plt.figure(figsize=(12, 5))

# Plot 1: CV scores vs depth
plt.subplot(1, 2, 1)
depths = list(depth_cv_scores.keys())
means = [depth_cv_scores[d]['mean'] for d in depths]
stds = [depth_cv_scores[d]['std'] for d in depths]

plt.errorbar(depths, means, yerr=stds, marker='o', capsize=5, capthick=2)
plt.xlabel('Tree Depth')
plt.ylabel('Cross-Validation Accuracy')
plt.title('5-Fold CV: Accuracy vs Tree Depth')
plt.grid(True, alpha=0.3)
plt.xticks(depths)

# Plot 2: Best depths distribution from nested CV
plt.subplot(1, 2, 2)
depth_counts = {d: best_depths.count(d) for d in set(best_depths)}
plt.bar(depth_counts.keys(), depth_counts.values(), alpha=0.7)
plt.xlabel('Tree Depth')
plt.ylabel('Frequency (Nested CV)')
plt.title('Optimal Depth Distribution (Nested CV)')
plt.grid(True, alpha=0.3)
plt.xticks(list(depth_counts.keys()))

plt.tight_layout()
plt.show()

# Print detailed results
print("\nRegular 5-Fold CV Results by Depth:")
print("-" * 40)
print(f"{'Depth':<8} {'Mean Acc':<12} {'Std Dev':<10}")
print("-" * 40)
for depth in depths:
    mean_acc = depth_cv_scores[depth]['mean']
    std_acc = depth_cv_scores[depth]['std']
    print(f"{depth:<8} {mean_acc:<12.4f} {std_acc:<10.4f}")

# Find optimal depth from regular CV
optimal_depth_regular = max(depth_cv_scores, key=lambda x: depth_cv_scores[x]['mean'])
print(f"\nOptimal depth (Regular CV): {optimal_depth_regular}")
print(f"Best CV score: {depth_cv_scores[optimal_depth_regular]['mean']:.4f}")

# Train final model with optimal depth
print(f"\nTraining final model with optimal depth: {optimal_depth_regular}")
final_dt = DecisionTree(criterion="information_gain", max_depth=optimal_depth_regular)
final_dt.fit(X_df, y_series)

# Plot final optimized tree
final_dt.plot(filename="optimized_classification_tree", view=False)

print("\n" + "="*60)
print("EXPERIMENT SUMMARY")
print("="*60)
print(f"Dataset: {len(X)} samples, {X.shape[1]} features, {len(np.unique(y))} classes")
print(f"70/30 Split Test Accuracy: {test_accuracy:.4f}")
print(f"Nested CV Mean Score: {np.mean(outer_scores):.4f} (+/- {np.std(outer_scores)*2:.4f})")
print(f"Optimal Depth (Nested CV): {max(set(best_depths), key=best_depths.count)}")
print(f"Optimal Depth (Regular CV): {optimal_depth_regular}")
print("="*60)