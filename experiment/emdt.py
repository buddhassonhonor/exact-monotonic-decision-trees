import numpy as np
from ortools.sat.python import cp_model
import time
from itertools import product
from math import prod

class ExactMonotonicTree:
    def __init__(
        self,
        max_depth=2,
        time_limit=300,
        monotonic=True,
        monotonic_features=None,
        num_workers=1,
        random_seed=42,
        warm_start_solution=None,
        monotonic_encoding='subtree',
        max_cells=4096,
        compress_rows=False,
        constant_hint=False,
    ):
        self.max_depth = max_depth
        self.time_limit = time_limit
        self.monotonic = monotonic
        self.monotonic_features = monotonic_features
        self.num_workers = num_workers
        self.random_seed = random_seed
        self.warm_start_solution = warm_start_solution
        if monotonic_encoding not in {'subtree', 'cells'}:
            raise ValueError('monotonic_encoding must be subtree or cells')
        self.monotonic_encoding = monotonic_encoding
        self.max_cells = int(max_cells)
        self.compress_rows = bool(compress_rows)
        self.constant_hint = bool(constant_hint)
        self.model = None
        self.solver = None
        self.solution_ = None
        self.fit_stats_ = {}

    def _resolve_split_index(self, feature_id, threshold):
        for k, (f, t) in enumerate(self.splits):
            if int(f) == int(feature_id) and abs(float(t) - float(threshold)) <= 1e-12:
                return k
        return None
        
    def fit(self, X, y):
        """
        Fits an exact decision tree using CP-SAT.
        Assumes X is normalized or we use pre-computed splits.
        For simplicity in this research prototype, we binarize X based on quantiles.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        if X.ndim != 2 or len(y) != len(X) or not np.all(np.isin(y, [0, 1])):
            raise ValueError('Expected a feature matrix and binary labels')
        if not np.isfinite(X).all() or self.max_depth < 1:
            raise ValueError('Finite features and depth >= 1 are required')
        if self.monotonic_features is not None and any(
                int(j) != j or not 0 <= int(j) < X.shape[1] for j in self.monotonic_features):
            raise ValueError('Governed feature indices are outside the feature matrix')
        y = y.astype(int)
        self.solution_ = None
        # 1. Preprocessing: Binarize features to simplify SAT encoding
        # We generate candidate splits (feature, threshold)
        self.n_samples, self.n_features = X.shape
        self.classes = np.unique(y)
        
        # Generate candidate splits (e.g., deciles)
        self.splits = []
        for f in range(self.n_features):
            unique_vals = np.unique(X[:, f])
            # Limit number of splits to keep problem size manageable
            if len(unique_vals) > 5:
                thresholds = np.percentile(X[:, f], np.linspace(20, 80, 4))
            else:
                thresholds = unique_vals[:-1]
            
            for t in thresholds:
                self.splits.append((f, t))
        
        # Compute binary feature matrix B where B[i, k] = 1 if X[i, f_k] > t_k
        self.n_splits = len(self.splits)
        if not self.n_splits:
            raise ValueError('At least one nonconstant feature is required')
        raw_n = self.n_samples
        class_counts = np.column_stack([1-y, y])
        if self.compress_rows:
            raw_B = np.column_stack([(X[:, f] > t) for f, t in self.splits])
            _, first, inverse = np.unique(raw_B, axis=0, return_index=True, return_inverse=True)
            class_counts = np.zeros((len(first), 2), dtype=int)
            np.add.at(class_counts, (inverse, y), 1)
            X = X[first]
            y = y[first]
            self.n_samples = len(first)
        self.class_counts_ = class_counts
        self.training_points_ = X
        routing_X = X
        self.cell_points_ = None
        if self.monotonic and self.monotonic_encoding == 'cells':
            if np.any(X < 0) or np.any(X > 1):
                raise ValueError('Cell encoding requires features in [0, 1]')
            axes = [sorted({float(t) for f, t in self.splits if f == j} | {1.0})
                    for j in range(self.n_features)]
            shape = tuple(len(axis) for axis in axes)
            n_cells = prod(shape)
            if n_cells > self.max_cells:
                raise ValueError(f'{n_cells} cells exceed max_cells={self.max_cells}')
            self.cell_points_ = np.asarray(list(product(*axes)), dtype=float)
            self.cell_shape_ = shape
            routing_X = np.vstack([X, self.cell_points_])
        n_routing = len(routing_X)
        self.B = np.zeros((n_routing, self.n_splits), dtype=int)
        for k, (f, t) in enumerate(self.splits):
            self.B[:, k] = (routing_X[:, f] > t).astype(int)
            
        # 2. Build CP-SAT Model
        model = cp_model.CpModel()
        
        # Tree Structure
        # Nodes 1..2^D-1 are internal, 2^D..2^{D+1}-1 are leaves
        n_internal = (1 << self.max_depth) - 1
        n_leaves = 1 << self.max_depth
        total_nodes = n_internal + n_leaves
        
        # Variables
        # a[n, k]: Node n splits on candidate split k
        a = {} 
        for n in range(1, n_internal + 1):
            for k in range(self.n_splits):
                a[n, k] = model.NewBoolVar(f'a_{n}_{k}')
            # Each internal node selects exactly one split
            model.Add(sum(a[n, k] for k in range(self.n_splits)) == 1)
            
        # leaf_class[l]: Prediction of leaf l (assumed binary 0/1 for now)
        # For regression or multi-class, this changes. Let's assume binary y in {0, 1}.
        c = {}
        for l in range(n_internal + 1, total_nodes + 1):
            c[l] = model.NewBoolVar(f'c_{l}')
            
        # z[i, l]: Sample i falls into leaf l
        z = {}
        for i in range(n_routing):
            for l in range(n_internal + 1, total_nodes + 1):
                z[i, l] = model.NewBoolVar(f'z_{i}_{l}')
            # Each sample in exactly one leaf
            model.Add(sum(z[i, l] for l in range(n_internal + 1, total_nodes + 1)) == 1)
            
        # Path Constraints
        # For each sample i and leaf l, if z[i, l] is true, then splits must match B
        for i in range(n_routing):
            for l in range(n_internal + 1, total_nodes + 1):
                # Trace path from root to l
                curr = l
                while curr > 1:
                    parent = curr // 2
                    is_right_child = (curr % 2 == 1)
                    
                    # If is_right_child, then split at parent must be true (X > t)
                    # i.e., sum_k a[parent, k] * B[i, k] == 1
                    # Since exactly one a[parent, k] is 1, we can say:
                    # z[i, l] => (Split satisfied)
                    
                    # Construct expression for "Split at parent is True for sample i"
                    # split_res = sum(a[parent, k] * B[i, k])
                    # We can't multiply variable by constant easily in boolean logic implied
                    
                    # Efficient encoding:
                    # z[i, l] implies sum(a[parent, k] for k where B[i, k] == required_val) == 1
                    
                    required_val = 1 if is_right_child else 0
                    valid_splits = [k for k in range(self.n_splits) if self.B[i, k] == required_val]
                    
                    if not valid_splits:
                        # Impossible path for this sample
                        model.Add(z[i, l] == 0)
                    else:
                        model.Add(sum(a[parent, k] for k in valid_splits) >= z[i, l])
                        
                    curr = parent

        # Objective: Minimize Error
        # error[i] = 1 if predicted class != true class
        # predicted class for i is sum(z[i, l] * c[l])
        # If y[i] = 1, we want sum(z[i, l] * c[l]) = 1.
        # If y[i] = 0, we want sum(z[i, l] * c[l]) = 0.
        
        errors = []
        predictions = []
        product_vars = {}
        for i in range(n_routing):
            # predicted_val = sum(z[i, l] AND c[l])
            # But z[i, l] and c[l] are bools.
            # Let p_i be the prediction for sample i.
            # p_i = sum_l (z[i,l] AND c[l])  (since only one z is 1)
            
            # We can introduce auxiliary var p[i]
            p_i = model.NewBoolVar(f'pred_{i}')
            
            # Link p_i to z and c
            # p_i <=> OR_l (z[i,l] AND c[l])
            # Implementation:
            # 1. If z[i,l]=1 and c[l]=1 => p_i=1
            # 2. If p_i=1 => sum(z[i,l] and c[l]) >= 1
            
            # Easier:
            # is_correct <=> (p_i == y[i])
            # Since y[i] is constant:
            # If y[i]==1: is_correct <=> p_i
            # If y[i]==0: is_correct <=> NOT p_i
            
            # Define product terms leaf_pred[i, l] = z[i, l] AND c[l]
            leaf_preds = []
            for l in range(n_internal + 1, total_nodes + 1):
                lp = model.NewBoolVar(f'lp_{i}_{l}')
                product_vars[i, l] = lp
                model.AddBoolAnd([z[i, l], c[l]]).OnlyEnforceIf(lp)
                model.AddBoolOr([z[i, l].Not(), c[l].Not()]).OnlyEnforceIf(lp.Not())
                leaf_preds.append(lp)
            
            model.Add(p_i == sum(leaf_preds))
            predictions.append(p_i)
            
            if i >= self.n_samples:
                continue
            errors.append(int(class_counts[i, 0])*p_i + int(class_counts[i, 1])*(1-p_i))
                
        model.Minimize(sum(errors))
        
        # 3. Monotonicity Constraints
        if self.monotonic:
            if self.monotonic_features is None:
                mono_features = set(range(self.n_features))
            else:
                mono_features = set(int(f) for f in self.monotonic_features)
            mono_split_idx = [k for k, (f, _) in enumerate(self.splits) if f in mono_features]

            # Get descendants helper
            def get_leaves(node):
                if node >= n_internal + 1:
                    return [node]
                return get_leaves(2*node) + get_leaves(2*node + 1)
            
            # Implementation of Local Monotonicity:
            # For each internal node n, enforce that all left leaves <= all right leaves
            for n in range(1, n_internal + 1) if self.monotonic_encoding == 'subtree' else []:
                l_leaves = get_leaves(2*n)
                r_leaves = get_leaves(2*n + 1)
                if not mono_split_idx:
                    continue

                mono_active = sum(a[n, k] for k in mono_split_idx)
                for ll in l_leaves:
                    for lr in r_leaves:
                        # Enforce ordering only when node n selects a monotonic feature.
                        model.Add(c[ll] <= c[lr] + 1 - mono_active)

            if self.monotonic_encoding == 'cells':
                for index in np.ndindex(self.cell_shape_):
                    q = self.n_samples + int(np.ravel_multi_index(index, self.cell_shape_))
                    for j in mono_features:
                        if index[j] + 1 < self.cell_shape_[j]:
                            upper = list(index)
                            upper[j] += 1
                            r = self.n_samples + int(np.ravel_multi_index(tuple(upper), self.cell_shape_))
                            model.Add(predictions[q] <= predictions[r])

        # Optional warm-start hints from a prior tree solution.
        hinted_splits = 0
        hinted_leaves = 0
        if isinstance(self.warm_start_solution, dict):
            structure = self.warm_start_solution.get('structure', {})
            leaves = self.warm_start_solution.get('leaves', {})
            for n, split in structure.items():
                try:
                    node_id = int(n)
                except (TypeError, ValueError):
                    continue
                if not (1 <= node_id <= n_internal):
                    continue
                if not isinstance(split, (tuple, list)) or len(split) != 2:
                    continue
                k = self._resolve_split_index(split[0], split[1])
                if k is None:
                    continue
                model.AddHint(a[node_id, k], 1)
                hinted_splits += 1

            for l, val in leaves.items():
                try:
                    leaf_id = int(l)
                except (TypeError, ValueError):
                    continue
                if leaf_id not in c:
                    continue
                model.AddHint(c[leaf_id], int(val))
                hinted_leaves += 1

        # A full feasible constant-tree hint uses training labels only and costs no solve.
        if self.constant_hint and self.warm_start_solution is None:
            constant = int(class_counts[:, 1].sum() >= class_counts[:, 0].sum())
            for n in range(1, n_internal + 1):
                for k in range(self.n_splits):
                    model.AddHint(a[n, k], int(k == 0))
            for leaf_id in c:
                model.AddHint(c[leaf_id], constant)
            for i in range(n_routing):
                node = 1
                while node <= n_internal:
                    node = 2*node + int(self.B[i, 0])
                for leaf_id in c:
                    selected = int(leaf_id == node)
                    model.AddHint(z[i, leaf_id], selected)
                    model.AddHint(product_vars[i, leaf_id], selected * constant)
                model.AddHint(predictions[i], constant)
            # This bound is valid for both encoded classes and does not remove an optimum.
            model.Add(sum(errors) <= int(min(class_counts[:, 0].sum(), class_counts[:, 1].sum())))

        # Solve
        solve_start = time.time()
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = self.time_limit
        solver.parameters.num_search_workers = self.num_workers
        solver.parameters.random_seed = self.random_seed
        status = solver.Solve(model)
        solve_end = time.time()
        
        self.status_ = status
        status_name = solver.StatusName(status)
        self.fit_stats_ = {
            'status': status_name,
            'time_limit': float(self.time_limit),
            'monotonic': bool(self.monotonic),
            'monotonic_encoding': self.monotonic_encoding,
            'n_cells': 0 if self.cell_points_ is None else len(self.cell_points_),
            'n_raw_training': raw_n,
            'n_training_patterns': self.n_samples,
            'compress_rows': self.compress_rows,
            'constant_hint': self.constant_hint,
            'n_monotonic_features': int(self.n_features if self.monotonic_features is None else len(self.monotonic_features)),
            'num_workers': int(self.num_workers),
            'random_seed': int(self.random_seed),
            'warm_start_used': bool(isinstance(self.warm_start_solution, dict)),
            'warm_start_hinted_splits': int(hinted_splits),
            'warm_start_hinted_leaves': int(hinted_leaves),
            'wall_time': float(solve_end - solve_start),
            'solver_wall_time': float(solver.WallTime())
        }
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            objective = float(solver.ObjectiveValue())
            bound = float(solver.BestObjectiveBound())
            if abs(objective) < 1e-12:
                gap = 0.0
            else:
                gap = max(0.0, (objective - bound) / abs(objective))
            self.fit_stats_.update({
                'objective': objective,
                'best_bound': bound,
                'relative_gap': gap
            })
            # Extract solution
            self.solution_ = {}
            self.solution_['structure'] = {}
            for n in range(1, n_internal + 1):
                for k in range(self.n_splits):
                    if solver.Value(a[n, k]):
                        self.solution_['structure'][n] = self.splits[k]
                        break
            
            self.solution_['leaves'] = {}
            for l in range(n_internal + 1, total_nodes + 1):
                self.solution_['leaves'][l] = solver.Value(c[l])
            self.routing_predictions_ = np.array([solver.Value(p) for p in predictions])
            self.model = model
            self.solver = solver
                
            return True
        else:
            self.fit_stats_.update({
                'objective': np.nan,
                'best_bound': np.nan,
                'relative_gap': np.nan
            })
            return False

    def predict(self, X):
        if not self.solution_:
            raise RuntimeError('No feasible tree is available; inspect fit_stats_')
            
        preds = []
        n_internal = (1 << self.max_depth) - 1
        
        for i in range(len(X)):
            curr = 1
            while curr <= n_internal:
                f, t = self.solution_['structure'].get(curr, (0, 0))
                if X[i, f] <= t:
                    curr = 2 * curr
                else:
                    curr = 2 * curr + 1
            
            preds.append(self.solution_['leaves'][curr])
            
        return np.array(preds)

    def check_monotonicity(self, X_grid):
        """
        Brute-force check on a grid.
        Returns number of violations.
        """
        preds = self.predict(X_grid)
        violations = 0
        total = 0
        n = len(X_grid)
        # Sample random pairs to check? Or full grid?
        # If n is small, full check.
        # Let's just check pairs (i, j) where X[i] <= X[j]
        
        # Optimization: Just check 1000 random pairs
        for _ in range(1000):
            i = np.random.randint(n)
            j = np.random.randint(n)
            if np.all(X_grid[i] <= X_grid[j]):
                if preds[i] > preds[j]:
                    violations += 1
            elif np.all(X_grid[j] <= X_grid[i]):
                if preds[j] > preds[i]:
                    violations += 1
        return violations
