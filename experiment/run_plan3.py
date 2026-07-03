import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
import time
from emdt import ExactMonotonicTree

# 1. Dataset Generation
def generate_credit_data(n_samples=500, noise=0.1, seed=42):
    np.random.seed(seed)
    # x1: Income (0-1), x2: Credit Score (0-1), x3: Debt Ratio (0-1, Reversed)
    X = np.random.rand(n_samples, 3)
    
    # True logic: Score = 0.4*Income + 0.4*CreditScore + 0.2*(1-Debt)
    # Monotonic in x1(+), x2(+), x3(-)
    # But for EMDT code we assume all monotonic increasing.
    # So we flip x3 input to (1-x3) "Repayment Capacity"
    X[:, 2] = 1 - X[:, 2] 
    
    score = 0.4*X[:, 0] + 0.4*X[:, 1] + 0.2*X[:, 2]
    y = (score > 0.5).astype(int)
    
    # Add noise: flip labels
    mask = np.random.rand(n_samples) < noise
    y[mask] = 1 - y[mask]
    
    return X, y

def count_violations(model, n_checks=1000, n_dim=3):
    X1 = np.random.rand(n_checks, n_dim)
    X2 = np.random.rand(n_checks, n_dim)
    X_min = np.minimum(X1, X2)
    X_max = np.maximum(X1, X2)
    p_min = model.predict(X_min)
    p_max = model.predict(X_max)
    return np.sum(p_min > p_max)

# 2. Rule Export
def export_cart_rules(tree, feature_names):
    tree_ = tree.tree_
    feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ]
    rules = []
    def recurse(node, path_str):
        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_name[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], path_str + f"If {name} <= {threshold:.2f}: ")
            recurse(tree_.children_right[node], path_str + f"If {name} > {threshold:.2f}: ")
        else:
            # Leaf
            val = tree_.value[node]
            cls = np.argmax(val)
            rules.append(f"{path_str} Predict {cls}")
    recurse(0, "")
    return rules

def export_emdt_rules(model, feature_names):
    if not model.solution_:
        return ["No solution found"]
    
    rules = []
    n_internal = (1 << model.max_depth) - 1
    
    def recurse(node, path_str):
        if node > n_internal:
            # Leaf
            cls = model.solution_['leaves'][node]
            rules.append(f"{path_str} Predict {cls}")
            return
        
        f, t = model.solution_['structure'].get(node, (0, 0))
        name = feature_names[f]
        recurse(2*node, path_str + f"If {name} <= {t:.2f}: ")
        recurse(2*node + 1, path_str + f"If {name} > {t:.2f}: ")
        
    recurse(1, "")
    return rules

def run_plan3():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(base_dir)
    figures_dir = os.path.join(project_root, 'figures')
    
    # ---------------------------------------------------------
    # Exp 1: Stability (Boxplots) with Credit Data
    # ---------------------------------------------------------
    print("Running Stability Experiment (10 seeds)...")
    results = []
    seeds = range(42, 52) # 10 seeds
    for seed in seeds:
        X, y = generate_credit_data(n_samples=300, noise=0.15, seed=seed)
        # Split
        split = int(0.8 * len(X))
        X_train, y_train = X[:split], y[:split]
        X_test, y_test = X[split:], y[split:]
        
        # CART
        cart = DecisionTreeClassifier(max_depth=3, random_state=seed)
        cart.fit(X_train, y_train)
        results.append({'Method': 'CART', 'Accuracy': accuracy_score(y_test, cart.predict(X_test)), 'Violations': count_violations(cart)})
        
        # EMDT
        emdt = ExactMonotonicTree(max_depth=3, monotonic=True, time_limit=30)
        emdt.fit(X_train, y_train)
        results.append({'Method': 'EMDT', 'Accuracy': accuracy_score(y_test, emdt.predict(X_test)), 'Violations': count_violations(emdt)})
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(base_dir, 'stability_results.csv'), index=False)
    
    # Boxplot
    plt.figure()
    df.boxplot(column='Accuracy', by='Method')
    plt.title('Accuracy Distribution (10 Seeds)')
    plt.suptitle('') # Remove auto-generated title
    plt.savefig(os.path.join(figures_dir, 'stability_accuracy.png'))
    plt.close()
    
    # ---------------------------------------------------------
    # Exp 2: Depth Ablation
    # ---------------------------------------------------------
    print("Running Depth Ablation...")
    ablation_res = []
    X, y = generate_credit_data(n_samples=300, noise=0.1, seed=42)
    for depth in [2, 3, 4]:
        emdt = ExactMonotonicTree(max_depth=depth, monotonic=True, time_limit=60)
        start = time.time()
        emdt.fit(X, y)
        runtime = time.time() - start
        acc = accuracy_score(y, emdt.predict(X)) # Train acc for simplicity
        ablation_res.append({'Depth': depth, 'Runtime': runtime, 'TrainAcc': acc})
        
    pd.DataFrame(ablation_res).to_csv(os.path.join(base_dir, 'ablation_results.csv'), index=False)
    
    # ---------------------------------------------------------
    # Exp 3: Case Study (Violation Instance)
    # ---------------------------------------------------------
    print("Generating Case Study (3D Violation Search)...")
    # Use full 3D data with Seed 51 (known to produce violations)
    X_case, y_case = generate_credit_data(n_samples=300, noise=0.15, seed=51) 
    
    cart = DecisionTreeClassifier(max_depth=3, random_state=51)
    cart.fit(X_case, y_case)
    
    # Find a concrete violation pair
    n_checks = 10000
    X1 = np.random.rand(n_checks, 3)
    X2 = np.random.rand(n_checks, 3)
    X_min = np.minimum(X1, X2) # The "Worse" profile
    X_max = np.maximum(X1, X2) # The "Better" profile
    
    p_min = cart.predict(X_min) # Should be <= p_max
    p_max = cart.predict(X_max)
    
    # Violation: Worse profile gets 1 (Approve), Better profile gets 0 (Reject)
    violation_indices = np.where(p_min > p_max)[0]
    
    with open(os.path.join(base_dir, 'case_study_violation.txt'), 'w') as f:
        if len(violation_indices) > 0:
            idx = violation_indices[0]
            v_worse = X_min[idx]
            v_better = X_max[idx]
            f.write(f"Violation Example (CART, Seed 51):\n")
            f.write(f"Applicant A (Worse Profile): Income={v_worse[0]:.2f}, Score={v_worse[1]:.2f}, RepaymentCap={v_worse[2]:.2f}\n")
            f.write(f"  -> Prediction: {p_min[idx]} (APPROVE)\n")
            f.write(f"Applicant B (Better Profile): Income={v_better[0]:.2f}, Score={v_better[1]:.2f}, RepaymentCap={v_better[2]:.2f}\n")
            f.write(f"  -> Prediction: {p_max[idx]} (REJECT)\n")
            f.write("\nConclusion: CART violates monotonicity constraints, rejecting a strictly better candidate.\n")
        else:
            f.write("No violation found in random checks.\n")
            
    print("Done.")

if __name__ == "__main__":
    run_plan3()
