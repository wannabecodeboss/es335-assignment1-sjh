import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree

np.random.seed(42)

# Function to create fake data for the 4 cases of decision trees
def create_fake_data(N, M, case_type):
    """
    Create fake data for different cases of decision trees
    Parameters:
    N: number of samples
    M: number of binary features
    case_type: 1, 2, 3, or 4 representing the four cases
    Returns:
    X: features DataFrame
    y: target Series
    """
    np.random.seed(42)  # For reproducibility
    # Generate M binary features
    X = pd.DataFrame(np.random.randint(0, 2, size=(N, M)),
                     columns=[f'feature_{i}' for i in range(M)])

    if case_type == 1:
        # Case 1: Discrete input, Discrete output (DIDO)
        y = pd.Series(np.random.randint(0, 3, size=N))  # 3 classes
    elif case_type == 2:
        # Case 2: Discrete input, Real output (DIRO)
        y = pd.Series(np.random.normal(50, 10, size=N))  # Normal distribution
    elif case_type == 3:
        # Case 3: Real input, Discrete output (RIDO)
        X = X.astype(float) + np.random.normal(0, 0.1, size=(N, M))  # Add noise to make real
        y = pd.Series(np.random.randint(0, 3, size=N))  # 3 classes
    elif case_type == 4:
        # Case 4: Real input, Real output (RIRO)
        X = X.astype(float) + np.random.normal(0, 0.1, size=(N, M))  # Add noise to make real
        y = pd.Series(np.random.normal(50, 10, size=N))  # Normal distribution

    return X, y

# Updated experiment parameters
N_values = list(range(1, 21))  # N from 1 to 100
M_values = list(range(1, 6))   # M from 1 to 10

def calculate_times_per_case(N_values, M_values, case_type, max_depth=5, num_runs=10):
    """Calculate averaged timing results for each (N, M) over multiple runs"""
    results = {'N_values': [], 'M_values': [], 'fit_times': [], 'predict_times': []}
    
    for N in N_values:
        for M in M_values:
            fit_times = []
            predict_times = []
            
            for _ in range(num_runs):
                # Create fresh data each run
                X_train, y_train = create_fake_data(N, M, case_type)
                X_test, y_test = create_fake_data(max(1, N//4), M, case_type)
                
                # Initialize decision tree
                criterion = "information_gain" if case_type in [1, 3] else "information_gain"
                dt = DecisionTree(criterion=criterion, max_depth=max_depth)
                
                # Time the fit operation
                start_time = time.time()
                dt.fit(X_train, y_train)
                fit_times.append(time.time() - start_time)
                
                # Time the predict operation
                start_time = time.time()
                _ = dt.predict(X_test)
                predict_times.append(time.time() - start_time)
            
            # Store averages
            results['N_values'].append(N)
            results['M_values'].append(M)
            results['fit_times'].append(np.mean(fit_times))
            results['predict_times'].append(np.mean(predict_times))
    
    return results

def plot_combined_results_with_theory(results, case_type, case_name):
    """Plot all 4 combinations in one figure with theoretical comparisons (smaller fonts + spacing)"""
    colors = ['blue', 'green', 'red', 'orange']
    color = colors[case_type-1]
    
    df = pd.DataFrame(results)
    N_fixed = int(np.median(df['N_values']))
    M_fixed = int(np.median(df['M_values']))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{case_name} - Combined Timing Analysis', fontsize=12, fontweight='bold')
    
    # 1. Training time vs M (fix N)
    df_N_fixed = df[df['N_values'] == N_fixed].sort_values('M_values')
    axes[0,0].plot(df_N_fixed['M_values'], df_N_fixed['fit_times'], 'o-', color=color, linewidth=2, markersize=6, label='Experimental')
    theoretical_train_M = df_N_fixed['M_values'] * N_fixed * np.log(np.maximum(N_fixed, 2))
    if df_N_fixed['fit_times'].iloc[0] > 0:
        theoretical_train_M = theoretical_train_M / theoretical_train_M.iloc[0] * df_N_fixed['fit_times'].iloc[0]
    axes[0,0].plot(df_N_fixed['M_values'], theoretical_train_M, 'r--', linewidth=2, label='Theoretical O(N*M*log(N))')
    axes[0,0].set_xlabel('Number of Features (M)', fontsize=10)
    axes[0,0].set_ylabel('Training Time (s)', fontsize=10)
    axes[0,0].set_title(f'Training vs M (N={N_fixed})', fontsize=11)
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=8)
    axes[0,0].tick_params(labelsize=8)
    
    # 2. Training time vs N (fix M)
    df_M_fixed = df[df['M_values'] == M_fixed].sort_values('N_values')
    axes[0,1].plot(df_M_fixed['N_values'], df_M_fixed['fit_times'], 'o-', color=color, linewidth=2, markersize=6, label='Experimental')
    theoretical_train_N = df_M_fixed['N_values'] * M_fixed * np.log(np.maximum(df_M_fixed['N_values'], 2))
    if df_M_fixed['fit_times'].iloc[0] > 0:
        theoretical_train_N = theoretical_train_N / theoretical_train_N.iloc[0] * df_M_fixed['fit_times'].iloc[0]
    axes[0,1].plot(df_M_fixed['N_values'], theoretical_train_N, 'r--', linewidth=2, label='Theoretical O(N*M*log(N))')
    axes[0,1].set_xlabel('Number of Samples (N)', fontsize=10)
    axes[0,1].set_ylabel('Training Time (s)', fontsize=10)
    axes[0,1].set_title(f'Training vs N (M={M_fixed})', fontsize=11)
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(fontsize=8)
    axes[0,1].tick_params(labelsize=8)
    
    # 3. Prediction time vs N
    axes[1,0].plot(df_M_fixed['N_values'], df_M_fixed['predict_times'], 'o-', color=color, linewidth=2, markersize=6, label='Experimental')
    theoretical_pred_N = np.log(np.maximum(df_M_fixed['N_values'], 2))
    if df_M_fixed['predict_times'].iloc[0] > 0:
        theoretical_pred_N = theoretical_pred_N / theoretical_pred_N.iloc[0] * df_M_fixed['predict_times'].iloc[0]
    axes[1,0].plot(df_M_fixed['N_values'], theoretical_pred_N, 'r--', linewidth=2, label='Theoretical O(log(N))')
    axes[1,0].set_xlabel('Number of Test Samples (N)', fontsize=10)
    axes[1,0].set_ylabel('Prediction Time (s)', fontsize=10)
    axes[1,0].set_title(f'Prediction vs N (M={M_fixed})', fontsize=11)
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend(fontsize=8)
    axes[1,0].tick_params(labelsize=8)
    
    # 4. Prediction time vs M
    axes[1,1].plot(df_N_fixed['M_values'], df_N_fixed['predict_times'], 'o-', color=color, linewidth=2, markersize=6, label='Experimental')
    theoretical_pred_M = np.ones_like(df_N_fixed['M_values']) * df_N_fixed['predict_times'].iloc[0] if df_N_fixed['predict_times'].iloc[0] > 0 else np.ones_like(df_N_fixed['M_values'])
    axes[1,1].plot(df_N_fixed['M_values'], theoretical_pred_M, 'r--', linewidth=2, label='Theoretical O(1)')
    axes[1,1].set_xlabel('Number of Features (M)', fontsize=10)
    axes[1,1].set_ylabel('Prediction Time (s)', fontsize=10)
    axes[1,1].set_title(f'Prediction vs M (N={N_fixed})', fontsize=11)
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(fontsize=8)
    axes[1,1].tick_params(labelsize=8)
    
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.35)
    plt.show()


def plot_summary_all_cases(all_results, case_names):
    """Create summary plots comparing all 4 tree types (smaller fonts + spacing)"""
    colors = ['blue', 'green', 'red', 'orange']
    dfs = {case: pd.DataFrame(res) for case, res in all_results.items()}
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle('Summary Comparison: All Decision Tree Types', fontsize=12, fontweight='bold')
    
    # 1. Training vs N
    for case_type, color in zip(all_results.keys(), colors):
        df = dfs[case_type]
        N_vals = sorted(df['N_values'].unique())
        avg_train_vs_N = [np.mean(df[df['N_values'] == N]['fit_times']) for N in N_vals]
        axes[0,0].plot(N_vals, avg_train_vs_N, 'o-', color=color, linewidth=2, label=f'{case_names[case_type]}')
        M_median = np.median(df['M_values'].unique())
        theoretical_train = np.array(N_vals) * M_median * np.log(np.maximum(np.array(N_vals), 2))
        if avg_train_vs_N[0] > 0:
            theoretical_train = theoretical_train / theoretical_train[0] * avg_train_vs_N[0]
        axes[0,0].plot(N_vals, theoretical_train, '--', color=color, alpha=0.5, linewidth=1)
    axes[0,0].set_xlabel('Number of Samples (N)', fontsize=10)
    axes[0,0].set_ylabel('Training Time (s)', fontsize=10)
    axes[0,0].set_title('Training vs N (avg over M)', fontsize=11)
    axes[0,0].set_yscale('log')
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].legend(fontsize=8)
    axes[0,0].tick_params(labelsize=8)
    
    # 2. Training vs M
    for case_type, color in zip(all_results.keys(), colors):
        df = dfs[case_type]
        M_vals = sorted(df['M_values'].unique())
        avg_train_vs_M = [np.mean(df[df['M_values'] == M]['fit_times']) for M in M_vals]
        axes[0,1].plot(M_vals, avg_train_vs_M, 'o-', color=color, linewidth=2, label=f'{case_names[case_type]}')
        N_median = np.median(df['N_values'].unique())
        theoretical_train = np.array(M_vals) * N_median * np.log(np.maximum(N_median, 2))
        if avg_train_vs_M[0] > 0:
            theoretical_train = theoretical_train / theoretical_train[0] * avg_train_vs_M[0]
        axes[0,1].plot(M_vals, theoretical_train, '--', color=color, alpha=0.5, linewidth=1)
    axes[0,1].set_xlabel('Number of Features (M)', fontsize=10)
    axes[0,1].set_ylabel('Training Time (s)', fontsize=10)
    axes[0,1].set_title('Training vs M (avg over N)', fontsize=11)
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].legend(fontsize=8)
    axes[0,1].tick_params(labelsize=8)
    
    # 3. Prediction vs N
    for case_type, color in zip(all_results.keys(), colors):
        df = dfs[case_type]
        N_vals = sorted(df['N_values'].unique())
        avg_pred_vs_N = [np.mean(df[df['N_values'] == N]['predict_times']) for N in N_vals]
        axes[1,0].plot(N_vals, avg_pred_vs_N, 'o-', color=color, linewidth=2, label=f'{case_names[case_type]}')
        theoretical_pred = np.log(np.maximum(np.array(N_vals), 2))
        if avg_pred_vs_N[0] > 0:
            theoretical_pred = theoretical_pred / theoretical_pred[0] * avg_pred_vs_N[0]
        axes[1,0].plot(N_vals, theoretical_pred, '--', color=color, alpha=0.5, linewidth=1)
    axes[1,0].set_xlabel('Number of Samples (N)', fontsize=10)
    axes[1,0].set_ylabel('Prediction Time (s)', fontsize=10)
    axes[1,0].set_title('Prediction vs N (avg over M)', fontsize=11)
    axes[1,0].set_yscale('log')
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].legend(fontsize=8)
    axes[1,0].tick_params(labelsize=8)
    
    # 4. Prediction vs M
    for case_type, color in zip(all_results.keys(), colors):
        df = dfs[case_type]
        M_vals = sorted(df['M_values'].unique())
        avg_pred_vs_M = [np.mean(df[df['M_values'] == M]['predict_times']) for M in M_vals]
        axes[1,1].plot(M_vals, avg_pred_vs_M, 'o-', color=color, linewidth=2, label=f'{case_names[case_type]}')
        theoretical_pred = np.ones_like(np.array(M_vals)) * (avg_pred_vs_M[0] if avg_pred_vs_M[0] > 0 else 1)
        axes[1,1].plot(M_vals, theoretical_pred, '--', color=color, alpha=0.5, linewidth=1)
    axes[1,1].set_xlabel('Number of Features (M)', fontsize=10)
    axes[1,1].set_ylabel('Prediction Time (s)', fontsize=10)
    axes[1,1].set_title('Prediction vs M (avg over N)', fontsize=11)
    axes[1,1].set_yscale('log')
    axes[1,1].grid(True, alpha=0.3)
    axes[1,1].legend(fontsize=8)
    axes[1,1].tick_params(labelsize=8)
    
    plt.tight_layout(pad=2.5)
    plt.subplots_adjust(top=0.90, hspace=0.4, wspace=0.35)
    plt.show()

# Main execution function
def run_complexity_experiments():
    """Run the complete runtime complexity analysis"""
    print("="*60)
    print("DECISION TREE RUNTIME COMPLEXITY EXPERIMENTS")
    print("Running experiments with N: 1-100, M: 1-10, 100 runs per combination")
    print("="*60)
    
    # Store results for all 4 cases
    all_results = {}
    case_names = {1: "DIDO (Discrete Input, Discrete Output)",
                  2: "DIRO (Discrete Input, Real Output)",
                  3: "RIDO (Real Input, Discrete Output)",
                  4: "RIRO (Real Input, Real Output)"}
    
    # Run experiments for each case in sequence
    for case_type in [1, 2, 3, 4]:
        print(f"\n{'='*80}")
        print(f"PROCESSING: {case_names[case_type]}")
        print(f"{'='*80}")
        
        # Calculate timing results
        results = calculate_times_per_case(N_values, M_values, case_type, max_depth=5)
        all_results[case_type] = results
        
        # Plot combined results with theoretical comparisons
        print(f"Generating combined plots for {case_names[case_type]}...")
        plot_combined_results_with_theory(results, case_type, case_names[case_type])
        
        print(f"COMPLETED: {case_names[case_type]}")
    
    # Generate summary comparison plots
    print(f"\n{'='*80}")
    print("GENERATING SUMMARY COMPARISON PLOTS")
    print(f"{'='*80}")
    plot_summary_all_cases(all_results, case_names)
    
    return all_results

# Run the experiments
if __name__ == "__main__":
    results = run_complexity_experiments()
    print("\nExperiments completed successfully!")
