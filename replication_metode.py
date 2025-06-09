import numpy as np

def generate_condind_dataset(n=10000, d=15, true_label_prob=0.5, seed=42):
    np.random.seed(seed)
    
    # Generate true labels (0 or 1)
    Y_true = np.random.binomial(1, true_label_prob, size=n)

    # Sensitivity and specificity for useful classifiers
    good_sens = 0.9  # P(pred=1 | Y=1)
    good_spec = 0.9  # P(pred=0 | Y=0)

    # Generate predictions from 5 good classifiers
    X_good = np.zeros((n, 5))
    for i in range(5):
        X_good[Y_true == 1, i] = np.random.binomial(1, good_sens, size=(Y_true == 1).sum())
        X_good[Y_true == 0, i] = np.random.binomial(1, 1 - good_spec, size=(Y_true == 0).sum())

    # Generate predictions from 10 random classifiers (accuracy ~ 50%)
    X_rand = np.random.binomial(1, 0.5, size=(n, 10))

    # Combine all classifiers
    X_total = np.hstack([X_good, X_rand])
    return X_total, Y_true

from sklearn.neural_network import BernoulliRBM
from sklearn.metrics import balanced_accuracy_score
from sklearn.linear_model import LogisticRegression

# Generate data
X, y_true = generate_condind_dataset()

# Train RBM with 1 hidden unit
rbm = BernoulliRBM(n_components=1, learning_rate=0.05, batch_size=10, n_iter=20, random_state=42)
rbm.fit(X)

# Get hidden activations (probability H=1 given X)
H = rbm.transform(X)  # shape (n_samples, 1)

# Use logistic regression to learn mapping H -> Y (mimicking EM)
clf = LogisticRegression()
clf.fit(H, y_true)
y_pred = clf.predict(H)

# Evaluate
score = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy: {score:.4f}")

import matplotlib.pyplot as plt

plt.hist(H, bins=50)
plt.title("RBM Hidden Unit Activation (Prob H=1)")
plt.xlabel("Activation")
plt.ylabel("Count")
plt.grid(True)
plt.show()
