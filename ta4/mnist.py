"""
MNIST - Classificação com k-NN e Classificador Linear
Exploração com diferentes splits e redução de dimensionalidade
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns

sns.set(style="whitegrid")

# 1. Carregar o dataset MNIST
print("Carregando o dataset MNIST...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target.astype(int)
print(f"Shape dos dados: {X.shape}, Labels: {y.shape}")

# 2. Funções auxiliares
def split_data(X, y, train_size, val_size, test_size, random_state=42):
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, train_size=train_size, random_state=random_state, stratify=y)
    val_relative_size = val_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=val_relative_size, random_state=random_state, stratify=y_temp)
    return X_train, X_val, X_test, y_train, y_val, y_test

def apply_pca(X_train, X_val, X_test, n_components=0.95):
    pca = PCA(n_components=n_components, whiten=True, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    print(f"PCA: de {X_train.shape[1]} para {X_train_pca.shape[1]} dimensões.")
    return X_train_pca, X_val_pca, X_test_pca

def train_knn(X_train, y_train, X_val, y_val, k, metric='euclidean'):
    knn = KNeighborsClassifier(n_neighbors=k, metric=metric, n_jobs=-1)
    knn.fit(X_train, y_train)
    y_val_pred = knn.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"k-NN (k={k}, metric={metric}): Validação: {acc:.4f}")
    return knn

def train_logistic(X_train, y_train, X_val, y_val):
    clf = LogisticRegression(max_iter=1000, solver='lbfgs', multi_class='multinomial', n_jobs=-1)
    clf.fit(X_train, y_train)
    y_val_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_val_pred)
    print(f"Logistic Regression: Validação: {acc:.4f}")
    return clf

# 3. Divisão dos dados e PCA
print("\nDividindo os dados...")
# Exemplo: 60% treino, 20% validação, 20% teste
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, 0.6, 0.2, 0.2)

print("Aplicando PCA...")
X_train_pca, X_val_pca, X_test_pca = apply_pca(X_train, X_val, X_test, n_components=0.95)

# 4. Experimentos com k-NN
print("\n=== Experimentos com k-NN ===")
for metric in ['euclidean', 'manhattan']:
    for k in [3, 5, 7]:
        knn = train_knn(X_train_pca, y_train, X_val_pca, y_val, k, metric)
        y_test_pred = knn.predict(X_test_pca)
        test_acc = accuracy_score(y_test, y_test_pred)
        print(f"Teste k-NN (k={k}, metric={metric}): {test_acc:.4f}\n")

# 5. Experimentos com Classificador Linear
print("\n=== Experimentos com Logistic Regression ===")
logistic = train_logistic(X_train_pca, y_train, X_val_pca, y_val)
y_test_pred = logistic.predict(X_test_pca)
test_acc = accuracy_score(y_test, y_test_pred)
print(f"Teste Logistic Regression: {test_acc:.4f}")

# 6. Avaliação final
print("\n=== Avaliação Final ===")
print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_test_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_test_pred))
