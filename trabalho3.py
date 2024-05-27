import numpy as np
from sklearn.decomposition import NMF, PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from skimage.feature import hog
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input
from tensorflow.keras.utils import to_categorical

# Carregamento dos dados
data = np.loadtxt('ocr_car_numbers_rotulado.txt', delimiter=' ')
X = data[:, :-1].reshape(-1, 35, 35)  # Atributos
y = data[:, -1]  # Rótulos

# Extração de atributos

def extract_hog(images):
    hog_features = []
    for img in images:
        feature = hog(img, pixels_per_cell = (4, 4), cells_per_block = (2, 2), feature_vector = True)
        hog_features.append(feature)
    return np.array(hog_features)

# Extração de Atributos
X_pca = PCA(n_components=50).fit_transform(X.reshape(len(X), -1))
X_lda = LDA(n_components=9).fit_transform(X.reshape(len(X), -1), y)
X_nmf = NMF(n_components=50, max_iter=1000).fit_transform(X.reshape(len(X), -1))
X_hog = extract_hog(X)


# Divisão em treino e teste
X_train_pca, X_test_pca, y_train, y_test = train_test_split(X_pca, y, test_size = 0.2, random_state = 42)
X_train_lda, X_test_lda = train_test_split(X_lda, test_size = 0.2, random_state = 42)
X_train_hog, X_test_hog = train_test_split(X_hog, test_size = 0.2, random_state = 42)
X_train_nmf, X_test_nmf = train_test_split(X_nmf, test_size = 0.2, random_state = 42)

# Modelos
classifiers = [
    ('SVM', SVC()),
    ('Random Forest', RandomForestClassifier()),
    ('Decision Tree', DecisionTreeClassifier()),
    ('Naive Bayes Gaussian', GaussianNB()),
    ('Gradient Boosting Machines', GradientBoostingClassifier()),
    
    # #('MLP', MLPClassifier())
    # ('MLP', MLPClassifier(hidden_layer_sizes=(100,), max_iter=500))
]

# Função para criar modelo de Deep Learning
def create_nn(input_shape):
    model = Sequential()
    model.add(Input(shape = input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(10, activation = 'softmax'))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

# Classe para integrar modelo Keras com scikit-learn
class KerasClassifierWrapper:
    def __init__(self, build_fn, epochs=10, batch_size=32, verbose=0):
        self.build_fn = build_fn
        self.epochs = epochs
        self.batch_size = batch_size
        self.verbose = verbose
        self.model = None
    
    def fit(self, X, y):
        self.model = self.build_fn()
        self.model.fit(X, y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbose)
    
    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return np.mean(y_pred == y)
    
    def get_params(self, deep=False):
        return {"build_fn": self.build_fn, "epochs": self.epochs, "batch_size": self.batch_size, "verbose": self.verbose}
    
    def set_params(self, **params):
        for param, value in params.items():
            setattr(self, param, value)
        return self

# Avaliação
results = {}
extractors = {
    'PCA': (X_train_pca, X_test_pca),
    'LDA': (X_train_lda, X_test_lda),
    'HOG': (X_train_hog, X_test_hog),
    'NMF': (X_train_nmf, X_test_nmf)
    # --- Espaco dos testes --- #
}

for extractor_name, (X_train, X_test) in extractors.items():
    results[extractor_name] = {}
    for clf_name, clf in classifiers:
        kfold = KFold(n_splits=10, shuffle=True, random_state=42)
        cv_results = cross_val_score(clf, X_train, y_train, cv=kfold, scoring='accuracy')
        results[extractor_name][clf_name] = {
            'Accuracy Mean': cv_results.mean(),
            'Accuracy Std': cv_results.std()
        }
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        results[extractor_name][clf_name]['Confusion Matrix'] = confusion_matrix(y_test, y_pred)
    
    # Deep Learning
    input_shape = X_train.shape[1:]
    model = KerasClassifierWrapper(build_fn=lambda: create_nn(input_shape), epochs=10, batch_size=32, verbose=0)
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results[extractor_name]['Neural Network'] = {
        'Accuracy Mean': cv_results.mean(),
        'Accuracy Std': cv_results.std()
    }
    model.fit(X_train, y_train)
    y_pred_nn = model.predict(X_test)
    results[extractor_name]['Neural Network']['Confusion Matrix'] = confusion_matrix(y_test, y_pred_nn)

# Exibindo resultados
for extractor_name, classifiers in results.items():
    print(f"\nExtractor: {extractor_name}")
    for clf_name, result in classifiers.items():
        print(f"{clf_name}:")
        if 'Accuracy Mean' in result:
            print(f"\tAccuracy Mean: {result['Accuracy Mean']}")
            print(f"\tAccuracy Std: {result['Accuracy Std']}")
        print(f"\tConfusion Matrix:\n{result['Confusion Matrix']}\n")
