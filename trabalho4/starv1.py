import shutil
import os
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
from  tensorflow import keras
import keras.utils as image
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop

def copy_image(source_path, destination_path):
    try:
        shutil.copy(source_path, destination_path)
        print("Deu bom 🥳")
    except FileNotFoundError:
        print("Cadê? 😒")
    except PermissionError:
        print("Sem Sudo Sem Sucesso 😡")
        
datagen = ImageDataGenerator(
    zca_whitening = True,
    rotation_range = 60,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    shear_range = 0.3,
    zoom_range = [0.6,1.4],
    channel_shift_range = 0.0,
    fill_mode = 'constant',
    cval = 0.0,
    horizontal_flip = True,
    vertical_flip = True,
    rescale = 1./255,
    data_format = None,
    validation_split = 0.01,
    interpolation_order = 1,
    dtype = None,
)

data_folder = r"Caminho"
train_folder = os.path.join(data_folder, "train")
augmented_folder = os.path.join(data_folder, "aug_data")
mapper_file = os.path.join(data_folder, "mapper.xlsx")
mapper_df = pd.read_excel(mapper_file)

if not os.path.exists(augmented_folder):
    os.makedirs(augmented_folder)

for filename in os.listdir(train_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        image_path = os.path.join(train_folder, filename)
        constellation_label = mapper_df[mapper_df["Constellation Image file"] == filename]["Constellation name"].values[0]

        constellation_folder = os.path.join(augmented_folder, constellation_label)

        if not os.path.exists(constellation_folder):
            os.makedirs(constellation_folder)
            new_image_path = os.path.join(constellation_folder, filename)
            copy_image(image_path,new_image_path)
        img = image.load_img(image_path,target_size = (300,300))
        img = image.img_to_array(img)
        padded_image = cv2.copyMakeBorder(img, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value = (0, 0, 0))
        
        i = 0
        input_batch = padded_image.reshape(1, 500, 500, 3)
        for output in datagen.flow( input_batch,batch_size = 1, save_to_dir = constellation_folder):
            i = i+1
            if i == 1000:
                break
            
model1 = Sequential()
model1.add(Conv2D(16, kernel_size = (3, 3), activation = 'tanh', input_shape = (500, 500, 3)))
model1.add(Conv2D(32, (3, 3), activation = 'tanh'))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(32, (3, 3), activation = 'tanh'))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(32, (3, 3), activation = 'tanh'))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Conv2D(32, (3, 3), activation = 'tanh'))
model1.add(MaxPooling2D(pool_size = (2, 2)))
model1.add(Dropout(0.25))
model1.add(Flatten())
model1.add(Dense(32, activation =' tanh'))
model1.add(Dropout(0.5))
model1.add(Dense(88, activation = 'softmax'))
model1.summary()

model2 = Sequential()
model2.add(Conv2D(32, (3, 3), activation='relu', input_shape = (500, 500, 3)))
model2.add(MaxPooling2D(pool_size = (2, 2)))
model2.add(Conv2D(64, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))
model2.add(Flatten())
model2.add(Dense(128, activation = 'relu'))
model2.add(Dense(88, activation = 'softmax'))

model3 = Sequential([
    
    Conv2D(16, (3,3), activation='relu', input_shape = (500, 500, 3)),
    MaxPooling2D(2, 2),
    
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    
    Flatten(),
    
    Dense(512, activation='relu'),
    Dense(88, activation='softmax')
])

train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_directory(
        r"train/directory",  
        target_size=(500, 500),  
        batch_size=88,
        
        class_mode='categorical')

model2.compile(
    loss='CategoricalCrossentropy',
    optimizer = RMSprop(learning_rate=0.001),
    metrics = ['accuracy'])

history = model2.fit(
    train_generator,
    steps_per_epoch = 10,  
    epochs = 15,
    verbose = 1
)