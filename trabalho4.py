import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import RMSprop

data_folder = os.path.join("images")
label_folder = os.path.join("labels.csv")

def load_test_data(test_folder, labels_file, target_size=(500, 500)):
    # Carregar o arquivo CSV com as labels
    labels_df = pd.read_csv(labels_file)
    
    # Lista para armazenar os caminhos das imagens e as labels correspondentes
    images = []
    labels = []
    
    # Iterar sobre as imagens na pasta de teste
    for filename in os.listdir(test_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Caminho completo para a imagem
            image_path = os.path.join(test_folder, filename)
            
            # Carregar a imagem e redimensioná-la
            img = image.load_img(image_path, target_size=target_size)
            img_array = image.img_to_array(img)
            
            # Adicionar a imagem ao conjunto de dados
            images.append(img_array)
            
            # Extrair a label correspondente do arquivo CSV
            label = labels_df[labels_df['filename'] == filename]['class'].values[0]
            labels.append(label)
    
    # Converter para arrays numpy
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels


model1 = Sequential()
model1.add(Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=(500, 500, 3)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Conv2D(32, (3, 3), activation='tanh'))
model1.add(MaxPooling2D(pool_size=(2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation='tanh'))
model1.add(Dropout(0.5))
model1.add(Dense(88, activation='softmax'))
model1.summary()

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Conv2D(64, (3, 3), activation='relu'))
model2.add(MaxPooling2D(pool_size=(2, 2)))
model2.add(Flatten())
model2.add(Dense(128, activation='relu'))
model2.add(Dense(88, activation='softmax'))

model3 = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(88, activation='softmax')
])


history = model2.fit(
    train_generator,
    steps_per_epoch=10,
    epochs=15,
    verbose=1
)

# Compilar os modelos
model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model2.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model3.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Avaliar os modelos nos dados de teste
results_model1 = model1.evaluate(test_images, test_labels)
results_model2 = model2.evaluate(test_images, test_labels)
results_model3 = model3.evaluate(test_images, test_labels)

# Imprimir os resultados
print("Resultados do modelo 1:")
print(f"Acurácia: {results_model1[1]}")
print(f"Perda: {results_model1[0]}")

print("Resultados do modelo 2:")
print(f"Acurácia: {results_model2[1]}")
print(f"Perda: {results_model2[0]}")

print("Resultados do modelo 3:")
print(f"Acurácia: {results_model3[1]}")
print(f"Perda: {results_model3[0]}")



