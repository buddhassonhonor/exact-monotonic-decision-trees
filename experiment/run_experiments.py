import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import time
from emdt import ExactMonotonicTree

def generate_data(dataset_type='linear', n_samples=200, noise=0.1, seed=42):
    np.random.seed(seed)
    if dataset_type == 'linear':
        X = np.random.rand(n_samples, 2)
        y_prob = (X[:, 0] + X[:, 1]) / 2.0
        y = (y_prob > 0.5).astype(int)
    elif dataset_type == 'nonlinear':
        X = np.random.rand(n_samples, 2)
        # Circle quadrant: x1^2 + x2^2 > r^2
        y_prob = (X[:, 0]**2 + X[:, 1]**2)
        y = (y_prob > 0.6).astype(int) # Threshold approx median
    elif dataset_type == 'high_dim':
        X = np.random.rand(n_samples, 5)
        # Only first 2 are relevant and monotonic
        y_prob = (X[:, 0] + X[:, 1]) / 2.0
        y = (y_prob > 0.5).astype(int)
    
    # Add noise
    mask = np.random.rand(n_samples) < noise
    y[mask] = 1 - y[mask]
    return X, y

def count_violations(model, n_checks=1000, n_dim=2):
    # Generate random pairs
    X1 = np.random.rand(n_checks, n_dim)
    X2 = np.random.rand(n_checks, n_dim)
    
    # Ensure X1 <= X2
    X_min = np.minimum(X1, X2)
    X_max = np.maximum(X1, X2)
    
    p_min = model.predict(X_min)
    p_max = model.predict(X_max)
    
    # Violation if p_min > p_max
    violations = np.sum(p_min > p_max)
    return violations

def run_experiments_plan2():
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    figures_dir = os.path.join(project_root, 'figures')
    results_path = os.path.join(base_dir, 'results_plan2.csv')
    scalability_path = os.path.join(base_dir, 'scalability.csv')
    
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
        
    results = []
    
    # 1. Main Experiment: Datasets x Noise x Seeds
    datasets = ['linear', 'nonlinear', 'high_dim']
    noise_levels = [0.1, 0.2]
    seeds = [42, 43, 44] # 3 seeds
    
    for ds_name in datasets:
        n_dim = 5 if ds_name == 'high_dim' else 2
        for noise in noise_levels:
            print(f"Running {ds_name} - Noise {noise}")
            for seed in seeds:
                X, y = generate_data(ds_name, n_samples=200, noise=noise, seed=seed)
                
                # Split
                indices = np.random.permutation(len(X))
                split = int(0.8 * len(X))
                X_train, y_train = X[indices[:split]], y[indices[:split]]
                X_test, y_test = X[indices[split:]], y[indices[split:]]
                
                # CART
                cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
                start = time.time()
                cart.fit(X_train, y_train)
                cart.fit_time = time.time() - start
                cart_acc = accuracy_score(y_test, cart.predict(X_test))
                cart_viol = count_violations(cart, n_dim=n_dim)
                
                results.append({
                    'Dataset': ds_name, 'Noise': noise, 'Seed': seed, 'Method': 'CART',
                    'Accuracy': cart_acc, 'Violations': cart_viol, 'Time': cart.fit_time
                })
                
                # EMDT
                emdt = ExactMonotonicTree(max_depth=3, monotonic=True, time_limit=30)
                start = time.time()
                emdt.fit(X_train, y_train)
                emdt.fit_time = time.time() - start
                emdt_acc = accuracy_score(y_test, emdt.predict(X_test))
                emdt_viol = count_violations(emdt, n_dim=n_dim)
                
                results.append({
                    'Dataset': ds_name, 'Noise': noise, 'Seed': seed, 'Method': 'EMDT',
                    'Accuracy': emdt_acc, 'Violations': emdt_viol, 'Time': emdt.fit_time
                })
                
                # ODT
                odt = ExactMonotonicTree(max_depth=3, monotonic=False, time_limit=30)
                start = time.time()
                odt.fit(X_train, y_train)
                odt.fit_time = time.time() - start
                odt_acc = accuracy_score(y_test, odt.predict(X_test))
                odt_viol = count_violations(odt, n_dim=n_dim)
                
                results.append({
                    'Dataset': ds_name, 'Noise': noise, 'Seed': seed, 'Method': 'ODT',
                    'Accuracy': odt_acc, 'Violations': odt_viol, 'Time': odt.fit_time
                })

    df = pd.DataFrame(results)
    df.to_csv(results_path, index=False)
    print("Main results saved.")
    
    # 2. Scalability Experiment
    print("Running Scalability Analysis...")
    scalability_results = []
    sample_sizes = [100, 200, 300, 400]
    for n in sample_sizes:
        X, y = generate_data('linear', n_samples=n, noise=0.1, seed=42)
        
        emdt = ExactMonotonicTree(max_depth=3, monotonic=True, time_limit=60)
        start = time.time()
        emdt.fit(X, y)
        runtime = time.time() - start
        
        scalability_results.append({'N': n, 'Time': runtime})
        print(f"N={n}, Time={runtime:.4f}s")
        
    df_scale = pd.DataFrame(scalability_results)
    df_scale.to_csv(scalability_path, index=False)
    
    # Plot Scalability
    plt.figure()
    plt.plot(df_scale['N'], df_scale['Time'], marker='o')
    plt.xlabel('Sample Size')
    plt.ylabel('Runtime (s)')
    plt.title('EMDT Scalability (Depth=3)')
    plt.grid(True)
    plt.savefig(os.path.join(figures_dir, 'scalability.png'))
    plt.close()

if __name__ == "__main__":
    run_experiments_plan2()
