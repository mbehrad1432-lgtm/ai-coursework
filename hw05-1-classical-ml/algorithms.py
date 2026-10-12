# Submitted model implementations with portfolio revisions; see README.
import numpy as np

def train_test_split_numpy(X, y, test_size=0.2, seed=42):
    if len(X) != len(y) or len(X) < 2 or not 0 < test_size < 1:
        raise ValueError("Need matching arrays, >=2 observations and 0<test_size<1")
    indices = np.random.default_rng(seed).permutation(len(X))
    n_test = max(1, min(len(X)-1, int(np.ceil(test_size*len(X)))))
    test, train = indices[:n_test], indices[n_test:]
    return X[train], X[test], y[train], y[test]

def paired_targets(y_true, y_pred):
    true, pred = np.asarray(y_true).reshape(-1), np.asarray(y_pred).reshape(-1)
    if len(true) == 0 or true.shape != pred.shape:
        raise ValueError("Targets must have the same nonzero length")
    return true, pred

class LinearRegressionGD:
    def __init__(self, lr=0.05, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weight = None  # shape: (n_features, 1)
        self.bias = None

    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples, 1)
        """
        y = np.asarray(y).reshape(-1, 1)
        if len(X) != len(y) or len(y) == 0:
            raise ValueError("Need matching nonempty X and y")
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weight = np.zeros((n_features, 1))
        self.bias = 0.0

        # Gradient descent loop
        for i in range(self.n_iters):
            # Compute predictions
            y_pred = np.dot(X, self.weight) + self.bias

            # Compute gradients dw and db
            dw = (2/n_samples) * np.dot(X.T, (y_pred - y))
            db = (2/n_samples) * np.sum(y_pred - y)

            # Update parameters
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        """
        X: numpy array of shape (n_samples, n_features)
        returns: predictions of shape (n_samples, 1)
        """
        return np.dot(X, self.weight) + self.bias

def mean_squared_error(y_true, y_pred):
    true, pred = paired_targets(y_true, y_pred)
    return float(np.mean((true-pred)**2))

def r2_score(y_true, y_pred):
    true, pred = paired_targets(y_true, y_pred)
    residual = float(np.sum((true-pred)**2))
    total = float(np.sum((true-true.mean())**2))
    if total == 0:
        return 1.0 if residual == 0 else 0.0
    return 1 - residual/total

class LogisticRegressionGD:
    def __init__(self, lr=0.1, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None  # shape: (n_features,)
        self.bias = None      # scalar

    def sigmoid(self, z):
        # Implement sigmoid function
        z = np.asarray(z, dtype=float)
        result = np.empty_like(z)
        positive = z >= 0
        result[positive] = 1 / (1 + np.exp(-z[positive]))
        exp_z = np.exp(z[~positive])
        result[~positive] = exp_z / (1 + exp_z)
        return result

    def fit(self, X, y):
        """
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        y = np.asarray(y).reshape(-1)
        if len(X) != len(y) or len(y) == 0 or not np.all(np.isin(y,[0,1])):
            raise ValueError("Need matching nonempty data and binary labels")
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for i in range(self.n_iters):
            # Compute linear model and predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute gradients
            dw = (1/n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1/n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict_proba(self, X):
        # Compute probabilities using sigmoid
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)

    def predict(self, X):
        """
        Return class labels 0 or 1
        """
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)

def accuracy_score(y_true, y_pred):
    true, pred = paired_targets(y_true,y_pred)
    return float(np.mean(true==pred))

def precision_score(y_true, y_pred):
    true, pred = paired_targets(y_true,y_pred)
    tp = np.sum((true==1)&(pred==1))
    denominator = np.sum(pred==1)
    return float(tp/denominator) if denominator else 0.0

def recall_score(y_true, y_pred):
    true, pred = paired_targets(y_true,y_pred)
    tp = np.sum((true==1)&(pred==1))
    denominator = np.sum(true==1)
    return float(tp/denominator) if denominator else 0.0

def make_moons_numpy(n_samples=300, noise=0.3, seed=42):
    rng = np.random.default_rng(seed)
    first = n_samples//2
    a = np.linspace(0,np.pi,first)
    b = np.linspace(0,np.pi,n_samples-first)
    X = np.vstack((np.c_[np.cos(a),np.sin(a)],np.c_[1-np.cos(b),.5-np.sin(b)]))
    y = np.r_[np.zeros(first,dtype=int),np.ones(n_samples-first,dtype=int)]
    X += rng.normal(0,noise,X.shape)
    order = rng.permutation(n_samples)
    return X[order],y[order]

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, *, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value  # for leaf node

    def fit(self, X, y):
        y = np.asarray(y).reshape(-1)
        if len(X) != len(y) or len(y) == 0:
            raise ValueError("Need matching nonempty X and y")
        self.tree = self._build_tree(X, y)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Stopping conditions
        if (depth >= self.max_depth) or (n_labels == 1) or (n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        best_feature, best_thresh, best_gini = None, None, 1.0
        X_left, X_right, y_left, y_right = None, None, None, None

        # For each feature
        for feature_index in range(n_features):
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_idxs = np.where(X[:, feature_index] <= threshold)[0]
                right_idxs = np.where(X[:, feature_index] > threshold)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue

                gini_left = self._gini(y[left_idxs])
                gini_right = self._gini(y[right_idxs])
                weighted_gini = (len(left_idxs) / n_samples) * gini_left + (len(right_idxs) / n_samples) * gini_right

                if weighted_gini < best_gini:
                    best_gini = weighted_gini
                    best_feature = feature_index
                    best_thresh = threshold
                    X_left, X_right = X[left_idxs, :], X[right_idxs, :]
                    y_left, y_right = y[left_idxs], y[right_idxs]

        if best_feature is None:
            leaf_value = self._most_common_label(y)
            return self.Node(value=leaf_value)

        # Recursively build left and right subtrees
        left_subtree = self._build_tree(X_left, y_left, depth + 1)
        right_subtree = self._build_tree(X_right, y_right, depth + 1)
        return self.Node(feature_index=best_feature, threshold=best_thresh, left=left_subtree, right=right_subtree)

    def _gini(self, y):
        """
        Compute Gini impurity for labels y
        """
        m = len(y)
        if m == 0:
            return 0
        probs = [np.sum(y == c) / m for c in np.unique(y)]
        return 1 - sum(p**2 for p in probs)

    def _most_common_label(self, y):
        """
        Return the most common label in y
        """
        vals, counts = np.unique(y, return_counts=True)
        return vals[np.argmax(counts)]

    def predict(self, X):
        """
        Predict class labels for samples in X
        """
        return np.array([self._traverse_tree(x, self.tree) for x in X])

    def _traverse_tree(self, x, node):
        if node.value is not None:
            return node.value
        if x[node.feature_index] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

def confusion_matrix(y_true, y_pred, labels=None):
    true, pred = paired_targets(y_true,y_pred)
    labels = np.unique(np.r_[true,pred]) if labels is None else np.asarray(labels)
    if len(np.unique(labels)) != len(labels) or not np.all(np.isin(np.r_[true,pred],labels)):
        raise ValueError("Labels must uniquely cover all true and predicted classes")
    return np.array([[np.sum((true==a)&(pred==b)) for b in labels] for a in labels],dtype=int)
