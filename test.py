import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from skimage.feature import graycomatrix, graycoprops, hog, local_binary_pattern, corner_harris, corner_peaks
from skimage.transform import resize
from skimage.measure import shannon_entropy
from scipy.ndimage import sobel, gaussian_gradient_magnitude
from scipy.stats import skew, kurtosis


# Função para carregar os dados
def load_data(file_path):
    data = np.loadtxt(file_path)
    X = data[:, :-1]  # Features
    y = data[:, -1]   # Labels
    return X, y

# Carregar os dados
file_path = "ocr_car_numbers_rotulado.txt"
X, y = load_data(file_path)

# Dividir os dados em conjunto de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Extrator de atributos: Histograma de gradientes orientados (HOG)
def extract_hog_features(X):
    hog_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        hog_feature = hog(img_reshaped, orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), block_norm='L2-Hys', visualize=False)
        hog_features.append(hog_feature)
    return np.array(hog_features)

# --------------------------------- Extratores ------------------------------------------- #
def extract_color_features(X):
    color_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        color_feature = np.mean(img_reshaped, axis=(0, 1))
        color_features.append(color_feature)
    return np.array(color_features)

def extract_shannon_entropy(X):
    entropy_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        entropy = shannon_entropy(img_reshaped)
        entropy_features.append(entropy)
    return np.array(entropy_features)

def extract_glcm_contrast(X):
    contrast_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        resized_img = resize(img_reshaped, (64, 64))
        glcm = graycomatrix((resized_img * 255).astype(np.uint8), [1], [0], 256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        contrast_features.append(contrast)
    return np.array(contrast_features)

def extract_local_binary_pattern(X):
    lbp_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        lbp = local_binary_pattern(img_reshaped, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 9))
        lbp_features.append(hist)
    return np.array(lbp_features)

def extract_sobel_features(X):
    sobel_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        sobel_x = sobel(img_reshaped, axis=0, mode='constant')
        sobel_y = sobel(img_reshaped, axis=1, mode='constant')
        sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
        sobel_mean = np.mean(sobel_mag)
        sobel_std = np.std(sobel_mag)
        sobel_features.append([sobel_mean, sobel_std])
    return np.array(sobel_features)

def extract_skew_kurtosis(X):
    skew_kurtosis_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        skewness = skew(img_reshaped.ravel())
        kurt = kurtosis(img_reshaped.ravel())
        skew_kurtosis_features.append([skewness, kurt])
    return np.array(skew_kurtosis_features)

def extract_gaussian_gradient_magnitude(X):
    gaussian_gradient_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        gradient_mag = gaussian_gradient_magnitude(img_reshaped, sigma=1)
        gaussian_gradient_features.append(gradient_mag.mean())
    return np.array(gaussian_gradient_features).reshape(-1, 1)

def extract_corner_features(X):
    corner_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        corners = corner_peaks(corner_harris(img_reshaped), min_distance=5)
        corner_features.append(len(corners))
    return np.array(corner_features).reshape(-1, 1)

def extract_pixel_mean_std(X):
    pixel_features = []
    for img in X:
        img_reshaped = img.reshape((35, 35))
        pixel_mean = img_reshaped.mean()
        pixel_std = img_reshaped.std()
        pixel_features.append([pixel_mean, pixel_std])
    return np.array(pixel_features)

# ---------------------- Classificadores ----------------------------------------------- #

# Classificador: Máquina de Vetores de Suporte (SVM)
def svm_classifier():
    return SVC(kernel='linear')

# Classificador: K-Vizinhos Mais Próximos (KNN)
def knn_classifier():
    return KNeighborsClassifier(n_neighbors=5)

def mlp_classifier():
    return MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)

# Classificador: Árvores de Decisão
def decision_tree_classifier():
    return DecisionTreeClassifier()

# Classificador: Random Forest
def random_forest_classifier():
    return RandomForestClassifier()

# Classificador: Gradient Boosting Machines
def gradient_boosting_classifier():
    return GradientBoostingClassifier()

# Classificador: Regressão Logística
def logistic_regression_classifier():
    return LogisticRegression(max_iter=1000)

# Classificador: Naive Bayes Gaussiano
def gaussian_nb_classifier():
    return GaussianNB()

# Função para avaliar o modelo usando k-fold cross-validation
def evaluate_model(model, X_train, y_train):
    kfold = KFold(n_splits=10, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=kfold)
    return scores.mean(), scores.std()

# Função para treinar e testar o modelo
def train_test_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, cm

# Função principal
def main():
    # Selecionar extratores de atributos e classificadores
    extractor_funcs = [extract_hog_features, extract_glcm_contrast, extract_sobel_features, extract_color_features, extract_local_binary_pattern, extract_skew_kurtosis]
    classifier_funcs = [svm_classifier, knn_classifier, mlp_classifier, decision_tree_classifier, random_forest_classifier, gradient_boosting_classifier, logistic_regression_classifier, gaussian_nb_classifier]

    # Loop sobre todas as combinações possíveis de extratores e classificadores
    results = []
    for extractor_func in extractor_funcs:
        for classifier_func in classifier_funcs:
            # Extrair atributos
            X_train_features = extractor_func(X_train)
            X_test_features = extractor_func(X_test)

            # Treinar e testar o modelo
            classifier = classifier_func()
            acc_mean, acc_std = evaluate_model(classifier, X_train_features, y_train)
            acc, cm = train_test_model(classifier, X_train_features, X_test_features, y_train, y_test)
            results.append((extractor_func.__name__, classifier_func.__name__, acc_mean, acc_std, acc, cm))

    # Imprimir os resultados
    for result in results:
        print("Extractor: {}, Classifier: {}, CV Accuracy Mean: {:.4f}, CV Accuracy Std: {:.4f}, Test Accuracy: {:.4f}".format(result[0], result[1], result[2], result[3], result[4]))
        print("Confusion Matrix:")
        print(result[5])

if __name__ == "__main__":
    main()


