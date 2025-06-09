from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Binarizer, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import numpy as np

# Load data
data = load_breast_cancer()
# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data.data)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, data.target, test_size=0.3, random_state=42)

# Daftar model (anggap sebagai "annotator")
models = [
    LogisticRegression(max_iter=500),
    RandomForestClassifier(),
    KNeighborsClassifier(),
    SVC(probability=True),
    GaussianNB()
]

# Buat output prediksi dari masing-masing model (binarized)
preds = []
for model in models:
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    binarized = Binarizer(threshold=0.5).fit_transform(probs.reshape(-1, 1)).flatten()
    preds.append(binarized)

# Combine jadi data matrix (annotator output)
X_annotator = np.vstack(preds).T

'''
# Ini mirip kayak replikasi tadi
rbm = BernoulliRBM(n_components=1, learning_rate=0.05, batch_size=10, n_iter=30, random_state=42)
rbm.fit(X_annotator)

# Hidden representation
H = rbm.transform(X_annotator)

# Belajar label (buat evaluasi)
clf = LogisticRegression()
clf.fit(H, y_test)
y_pred = clf.predict(H)

# Balanced Accuracy
score = balanced_accuracy_score(y_test, y_pred)
print(f"Balanced Accuracy (Real Dataset): {score:.4f}")
'''