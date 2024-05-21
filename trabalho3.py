import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.calibration import cross_val_predict
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.neural_network import MLPClassifier

# # Passo 1: Pré-processamento dos dados
# data = np.loadtxt('ocr_car_numbers_rotulado.txt', delimiter = ' ')
# X = data[:, :-1].reshape(-1, 35, 35)  # Atributos (imagens)
# y = data[:, -1]  # Rótulos

# # Passo 2: Extração de Atributos
# pca = PCA(n_components = 50) 
# X_pca = pca.fit_transform(X.reshape(len(X), -1))

# # Passo 3: Classificação de Padrões
# X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# # Classificadores
# classifiers = [
#     ('SVM', SVC()),
#     ('Random Forest', RandomForestClassifier())
# ]

# # Passo 4: Avaliação
# results = {}
# for name, clf in classifiers:
#     kfold = KFold(n_splits = 10, shuffle = True, random_state = 42)
#     cv_results = cross_val_score(clf, X_train, y_train, cv = kfold, scoring = 'accuracy')
#     results[name] = {
#         'Accuracy Mean': cv_results.mean(),
#         'Accuracy Std': cv_results.std()
#     }
#     clf.fit(X_train, y_train)
#     y_pred = clf.predict(X_test)
#     results[name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)

# # Exibindo resultados
# for name, result in results.items():
#     print(f"{name}:")
#     print(f"\tAccuracy Mean: {result['Accuracy Mean']}")
#     print(f"\tAccuracy Std: {result['Accuracy Std']}")
#     print(f"\tConfusion Matrix:\n{result['Confusion Matrix']}\n")

# ------------------------------------------------------- #

data = pd.read_csv('ocr_car_numbers_rotulado.txt', header=None)
X = data.iloc[:, :-1].values  # Features
y = data.iloc[:, -1].values   # Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class IdentityTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X



# Definindo os extratores de atributos
pca = PCA(n_components=0.95)  # Manter 95% da variância explicada
# selectkbest = SelectKBest(f_classif, k=20)  # Selecionar as 20 melhores características
selectkbest = SelectKBest(f_classif)
scaler = StandardScaler()  # Normalização dos dados

# Definindo os classificadores
svm_classifier = SVC(kernel='linear', C=1)
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# pipelines = [
#     ("PCA + SVM", Pipeline(
#         [("pca", pca),
#         ("svm", svm_classifier)]
#     )),
#     ("SelectKBest + SVM", Pipeline(
#         [("selectkbest", selectkbest),
#         ("svm", svm_classifier)]
#     )),
#     ("StandardScaler + SVM", Pipeline(
#         [("scaler", scaler),
#         ("svm", svm_classifier)]
#     )),
#     ("PCA + Random Forest", Pipeline(
#         [("pca", pca),
#         ("rf", rf_classifier)]
#     )),
#     ("SelectKBest + Random Forest", Pipeline(
#         [("selectkbest", selectkbest),
#         ("rf", rf_classifier)]
#     )),
#     ("StandardScaler + Random Forest", Pipeline(
#         [("scaler", scaler),
#         ("rf", rf_classifier)]
#     ))
# ]

# Definindo os pipelines
pipelines = [
    ("PCA + SVM", Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("svm", svm_classifier)
    ])),
    ("SelectKBest + SVM", Pipeline([
        ("scaler", scaler),
        ("selectkbest", selectkbest),
        ("svm", svm_classifier)
    ])),
    ("StandardScaler + SVM", Pipeline([
        ("scaler", scaler),
        ("svm", svm_classifier)
    ])),
    ("PCA + Random Forest", Pipeline([
        ("scaler", scaler),
        ("pca", pca),
        ("rf", rf_classifier)
    ])),
    ("SelectKBest + Random Forest", Pipeline([
        ("scaler", scaler),
        ("selectkbest", selectkbest),
        ("rf", rf_classifier)
    ])),
    ("StandardScaler + Random Forest", Pipeline([
        ("scaler", scaler),
        ("rf", rf_classifier)
    ]))
]

results = {}
for name, pipeline in pipelines:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(pipeline, X_train, y_train, cv=kfold, scoring='accuracy')
    results[name] = cv_results

for name, scores in results.items():
    print(f"{name}: Mean Accuracy: {scores.mean()}, Standard Deviation: {scores.std()}")

# Para obter a matriz de confusão média
for name, pipeline in pipelines:
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    y_pred = cross_val_predict(pipeline, X_train, y_train, cv=kfold)
    cm = confusion_matrix(y_train, y_pred)
    print(f"{name}: Confusion Matrix:")
    print(cm / cm.sum(axis=1)[:, np.newaxis])  # Normalizando a matriz de confusão

