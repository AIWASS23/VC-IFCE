import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D

from keras import Sequential
from keras._tf_keras import Dense, Flatten, Conv2D, MaxPooling2D


# Passo 1: Pré-processamento dos dados
data = np.loadtxt('ocr_car_numbers_rotulado.txt', delimiter = ' ')
X = data[:, :-1].reshape(-1, 35, 35)  # Atributos (imagens)
y = data[:, -1]  # Rótulos

# Passo 2: Extração de Atributos
# Por exemplo, PCA
pca = PCA(n_components = 50)  # Reduzindo para 50 componentes
X_pca = pca.fit_transform(X.reshape(len(X), -1))

# Passo 3: Classificação de Padrões
# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Classificadores
classifiers = [
    ('SVM', SVC()),
    ('Random Forest', RandomForestClassifier())
]

# Passo 4: Avaliação
results = {}
for name, clf in classifiers:
    kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
    cv_results = cross_val_score(clf, X_train, y_train, cv = kfold, scoring = 'accuracy')
    results[name] = {
        'Accuracy Mean': cv_results.mean(),
        'Accuracy Std': cv_results.std()
    }
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    results[name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)

# Exibindo resultados
for name, result in results.items():
    print(f"{name}:")
    print(f"\tAccuracy Mean: {result['Accuracy Mean']}")
    print(f"\tAccuracy Std: {result['Accuracy Std']}")
    print(f"\tConfusion Matrix:\n{result['Confusion Matrix']}\n")
