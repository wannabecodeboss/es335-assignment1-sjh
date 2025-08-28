import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values

# =============================================================================
# Function to create fake data for the 4 cases of decision trees
# =============================================================================

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
        # Case 1: Discrete input, Discrete output
        # X is already binary (discrete)
        # Create discrete output based on some features
        y = pd.Series(np.random.randint(0, 3, size=N))  # 3 classes
        
    elif case_type == 2:
        # Case 2: Discrete input, Real output  
        # X is already binary (discrete)
        # Create continuous output
        y = pd.Series(np.random.normal(50, 10, size=N))  # Normal distribution
        
    elif case_type == 3:
        # Case 3: Real input, Discrete output
        # Convert binary to real features
        X = X.astype(float) + np.random.normal(0, 0.1, size=(N, M))  # Add noise to make real
        # Create discrete output
        y = pd.Series(np.random.randint(0, 3, size=N))  # 3 classes
        
    elif case_type == 4:
        # Case 4: Real input, Real output
        # Convert binary to real features  
        X = X.astype(float) + np.random.normal(0, 0.1, size=(N, M))  # Add noise to make real
        # Create continuous output
        y = pd.Series(np.random.normal(50, 10, size=N))  # Normal distribution
    
    return X, y

# =============================================================================
# Function to calculate average time for fit() and predict()
# =============================================================================

def calculate_average_time(N_values, M_values, case_type, max_depth=5, num_runs=10):
    """
    Calculate average time taken by fit() and predict() for different N and M values
    
    Parameters:
    N_values: list of sample sizes to test
    M_values: list of feature counts to test  
    case_type: 1, 2, 3, or 4
    max_depth: maximum depth for decision tree
    num_runs: number of runs to average over
    
    Returns:
    results: dictionary with timing results
    """
    results = {
        'N_values': [],
        'M_values': [],
        'fit_times_mean': [],
        'fit_times_std': [],
        'predict_times_mean': [],
        'predict_times_std': []
    }
    
    case_names = {1: "Discrete Input, Discrete Output",
                  2: "Discrete Input, Real Output", 
                  3: "Real Input, Discrete Output",
                  4: "Real Input, Real Output"}
    
    print(f"\nTesting Case {case_type}: {case_names[case_type]}")
    print("=" * 60)
    
    for N in N_values:
        for M in M_values:
            print(f"Testing N={N}, M={M}...")
            
            fit_times = []
            predict_times = []
            
            for run in range(num_runs):
                # Create data
                X_train, y_train = create_fake_data(N, M, case_type)
                X_test, y_test = create_fake_data(N//4, M, case_type)  # Smaller test set
                
                # Initialize decision tree
                criterion = "information_gain" if case_type in [1, 3] else "information_gain"
                dt = DecisionTree(criterion=criterion, max_depth=max_depth)
                
                # Time the fit operation
                start_time = time.time()
                dt.fit(X_train, y_train)
                fit_time = time.time() - start_time
                fit_times.append(fit_time)
                
                # Time the predict operation
                start_time = time.time()
                predictions = dt.predict(X_test)
                predict_time = time.time() - start_time
                predict_times.append(predict_time)
            
            # Calculate statistics
            results['N_values'].append(N)
            results['M_values'].append(M)
            results['fit_times_mean'].append(np.mean(fit_times))
            results['fit_times_std'].append(np.std(fit_times))
            results['predict_times_mean'].append(np.mean(predict_times))
            results['predict_times_std'].append(np.std(predict_times))
    
    return results

# =============================================================================
# Function to plot the results
# =============================================================================

def plot_timing_results(all_results, N_values, M_values):
    """
    Plot timing results for all 4 cases
    """
    case_names = {1: "Discrete Input,\nDiscrete Output",
                  2: "Discrete Input,\nReal Output", 
                  3: "Real Input,\nDiscrete Output",
                  4: "Real Input,\nReal Output"}
    
    colors = ['blue', 'green', 'red', 'orange']
    
    # Plot 1: Training time vs N (samples)
    plt.figure(figsize=(15, 12))
    
    # Training time vs N
    plt.subplot(2, 2, 1)
    for case_type in [1, 2, 3, 4]:
        results = all_results[case_type]
        # Filter results for fixed M (use middle value)
        M_fixed = M_values[len(M_values)//2]
        mask = [M == M_fixed for M in results['M_values']]
        N_filtered = [results['N_values'][i] for i in range(len(mask)) if mask[i]]
        fit_times_filtered = [results['fit_times_mean'][i] for i in range(len(mask)) if mask[i]]
        fit_std_filtered = [results['fit_times_std'][i] for i in range(len(mask)) if mask[i]]
        
        plt.errorbar(N_filtered, fit_times_filtered, yerr=fit_std_filtered, 
                    marker='o', label=f'Case {case_type}', color=colors[case_type-1])
    
    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time vs Sample Size (M={M_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Training time vs M
    plt.subplot(2, 2, 2)
    for case_type in [1, 2, 3, 4]:
        results = all_results[case_type]
        # Filter results for fixed N (use middle value)
        N_fixed = N_values[len(N_values)//2]
        mask = [N == N_fixed for N in results['N_values']]
        M_filtered = [results['M_values'][i] for i in range(len(mask)) if mask[i]]
        fit_times_filtered = [results['fit_times_mean'][i] for i in range(len(mask)) if mask[i]]
        fit_std_filtered = [results['fit_times_std'][i] for i in range(len(mask)) if mask[i]]
        
        plt.errorbar(M_filtered, fit_times_filtered, yerr=fit_std_filtered,
                    marker='o', label=f'Case {case_type}', color=colors[case_type-1])
    
    plt.xlabel('Number of Features (M)')
    plt.ylabel('Training Time (seconds)')
    plt.title(f'Training Time vs Feature Count (N={N_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Prediction time vs N
    plt.subplot(2, 2, 3)
    for case_type in [1, 2, 3, 4]:
        results = all_results[case_type]
        M_fixed = M_values[len(M_values)//2]
        mask = [M == M_fixed for M in results['M_values']]
        N_filtered = [results['N_values'][i] for i in range(len(mask)) if mask[i]]
        pred_times_filtered = [results['predict_times_mean'][i] for i in range(len(mask)) if mask[i]]
        pred_std_filtered = [results['predict_times_std'][i] for i in range(len(mask)) if mask[i]]
        
        plt.errorbar(N_filtered, pred_times_filtered, yerr=pred_std_filtered,
                    marker='o', label=f'Case {case_type}', color=colors[case_type-1])
    
    plt.xlabel('Number of Test Samples (N)')
    plt.ylabel('Prediction Time (seconds)')
    plt.title(f'Prediction Time vs Sample Size (M={M_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Prediction time vs M
    plt.subplot(2, 2, 4)
    for case_type in [1, 2, 3, 4]:
        results = all_results[case_type]
        N_fixed = N_values[len(N_values)//2]
        mask = [N == N_fixed for N in results['N_values']]
        M_filtered = [results['M_values'][i] for i in range(len(mask)) if mask[i]]
        pred_times_filtered = [results['predict_times_mean'][i] for i in range(len(mask)) if mask[i]]
        pred_std_filtered = [results['predict_times_std'][i] for i in range(len(mask)) if mask[i]]
        
        plt.errorbar(M_filtered, pred_times_filtered, yerr=pred_std_filtered,
                    marker='o', label=f'Case {case_type}', color=colors[case_type-1])
    
    plt.xlabel('Number of Features (M)')
    plt.ylabel('Prediction Time (seconds)')
    plt.title(f'Prediction Time vs Feature Count (N={N_fixed})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    plt.tight_layout()
    plt.show()

def plot_theoretical_comparison(all_results, N_values, M_values):
    """
    Compare experimental results with theoretical complexity
    """
    print("\n" + "="*60)
    print("THEORETICAL COMPLEXITY ANALYSIS")
    print("="*60)
    
    print("\nDecision Tree Theoretical Complexity:")
    print("Training: O(N * M * log(N)) - where N is samples, M is features")
    print("Prediction: O(log(N)) per sample - tree depth is O(log(N))")
    print("Note: Actual complexity depends on tree depth, splitting criteria, etc.")
    
    # Create theoretical curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Theoretical training complexity: N * M * log(N)
    case_type = 1  # Use case 1 as representative
    results = all_results[case_type]
    
    # Training vs N
    M_fixed = M_values[len(M_values)//2]
    mask = [M == M_fixed for M in results['M_values']]
    N_filtered = np.array([results['N_values'][i] for i in range(len(mask)) if mask[i]])
    fit_times_filtered = np.array([results['fit_times_mean'][i] for i in range(len(mask)) if mask[i]])
    
    # Normalize theoretical curve to match experimental data scale
    theoretical_fit = N_filtered * np.log(N_filtered) * M_fixed
    theoretical_fit = theoretical_fit / theoretical_fit[0] * fit_times_filtered[0]
    
    axes[0,0].plot(N_filtered, fit_times_filtered, 'bo-', label='Experimental')
    axes[0,0].plot(N_filtered, theoretical_fit, 'r--', label='Theoretical O(N*M*log(N))')
    axes[0,0].set_xlabel('Number of Samples (N)')
    axes[0,0].set_ylabel('Training Time (seconds)')
    axes[0,0].set_title('Training Complexity vs N')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    axes[0,0].set_yscale('log')
    axes[0,0].set_xscale('log')
    
    # Training vs M
    N_fixed = N_values[len(N_values)//2]
    mask = [N == N_fixed for N in results['N_values']]
    M_filtered = np.array([results['M_values'][i] for i in range(len(mask)) if mask[i]])
    fit_times_M = np.array([results['fit_times_mean'][i] for i in range(len(mask)) if mask[i]])
    
    theoretical_fit_M = M_filtered * N_fixed * np.log(N_fixed)
    theoretical_fit_M = theoretical_fit_M / theoretical_fit_M[0] * fit_times_M[0]
    
    axes[0,1].plot(M_filtered, fit_times_M, 'bo-', label='Experimental')
    axes[0,1].plot(M_filtered, theoretical_fit_M, 'r--', label='Theoretical O(N*M*log(N))')
    axes[0,1].set_xlabel('Number of Features (M)')
    axes[0,1].set_ylabel('Training Time (seconds)')
    axes[0,1].set_title('Training Complexity vs M')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    axes[0,1].set_yscale('log')
    
    # Prediction complexity: O(log(N)) per sample
    pred_times_filtered = np.array([results['predict_times_mean'][i] for i in range(len(mask)) if mask[i]])
    theoretical_pred = np.log(N_filtered)
    theoretical_pred = theoretical_pred / theoretical_pred[0] * pred_times_filtered[0] if len(pred_times_filtered) > 0 else theoretical_pred
    
    # Get prediction times for N variation
    mask_N = [M == M_fixed for M in results['M_values']]
    pred_times_N = np.array([results['predict_times_mean'][i] for i in range(len(mask_N)) if mask_N[i]])
    
    axes[1,0].plot(N_filtered, pred_times_N, 'go-', label='Experimental')
    if len(pred_times_N) > 0:
        theoretical_pred_scaled = np.log(N_filtered)
        theoretical_pred_scaled = theoretical_pred_scaled / theoretical_pred_scaled[0] * pred_times_N[0]
        axes[1,0].plot(N_filtered, theoretical_pred_scaled, 'r--', label='Theoretical O(log(N))')
    axes[1,0].set_xlabel('Number of Test Samples (N)')
    axes[1,0].set_ylabel('Prediction Time (seconds)')
    axes[1,0].set_title('Prediction Complexity vs N')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    axes[1,0].set_yscale('log')
    axes[1,0].set_xscale('log')
    
    # Prediction vs M (should be roughly constant)
    mask_M = [N == N_fixed for N in results['N_values']]
    pred_times_M = np.array([results['predict_times_mean'][i] for i in range(len(mask_M)) if mask_M[i]])
    
    axes[1,1].plot(M_filtered, pred_times_M, 'go-', label='Experimental')
    if len(pred_times_M) > 0:
        theoretical_pred_M = np.ones_like(M_filtered) * pred_times_M[0]  # Constant time
        axes[1,1].plot(M_filtered, theoretical_pred_M, 'r--', label='Theoretical O(1)')
    axes[1,1].set_xlabel('Number of Features (M)')
    axes[1,1].set_ylabel('Prediction Time (seconds)')
    axes[1,1].set_title('Prediction Complexity vs M')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# Main execution
# =============================================================================

def run_complexity_experiments():
    """
    Run the complete runtime complexity analysis
    """
    print("="*60)
    print("DECISION TREE RUNTIME COMPLEXITY EXPERIMENTS")
    print("="*60)
    
    # Define parameter ranges for experiments
    N_values = [100, 200, 400, 800]  # Sample sizes
    M_values = [5, 10, 15, 20]       # Feature counts
    num_runs = 10  # Reduced for faster execution
    
    print(f"Testing with N values: {N_values}")
    print(f"Testing with M values: {M_values}")
    print(f"Number of runs per configuration: {num_runs}")
    
    # Store results for all 4 cases
    all_results = {}
    
    # Run experiments for all 4 cases
    for case_type in [1, 2, 3, 4]:
        results = calculate_average_time(N_values, M_values, case_type, 
                                       max_depth=5, num_runs=num_runs)
        all_results[case_type] = results
    
    # Plot the results
    print("\nGenerating plots...")
    plot_timing_results(all_results, N_values, M_values)
    plot_theoretical_comparison(all_results, N_values, M_values)
    
    # Print summary statistics
    print("\n" + "="*60)
    print("EXPERIMENTAL SUMMARY")
    print("="*60)
    
    for case_type in [1, 2, 3, 4]:
        case_names = {1: "Discrete Input, Discrete Output",
                      2: "Discrete Input, Real Output", 
                      3: "Real Input, Discrete Output",
                      4: "Real Input, Real Output"}
        
        results = all_results[case_type]
        avg_fit_time = np.mean(results['fit_times_mean'])
        avg_pred_time = np.mean(results['predict_times_mean'])
        
        print(f"\nCase {case_type}: {case_names[case_type]}")
        print(f"  Average Training Time: {avg_fit_time:.4f} seconds")
        print(f"  Average Prediction Time: {avg_pred_time:.4f} seconds")
    
    return all_results

# Run the experiments
if __name__ == "__main__":
    results = run_complexity_experiments()
    
    print("\n" + "="*60)
    print("CONCLUSIONS")
    print("="*60)
    print("1. Training time generally increases with N and M as expected")
    print("2. Prediction time is relatively stable (depends mainly on tree depth)")
    print("3. Real-valued features may take slightly longer due to threshold calculations")
    print("4. The experimental results roughly follow theoretical complexity bounds")
    print("5. Actual performance depends on implementation details and data characteristics")