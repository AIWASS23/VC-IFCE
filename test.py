import pandas as pd
import os
import shutil
import time

# Juntar os csv

df1 = pd.read_csv('Constellation/train/annotations_train.csv', usecols=['filename', 'class'])
df2 = pd.read_csv('Constellation/test/annotations_test.csv', usecols=['filename', 'class'])
df3 = pd.read_csv('Constellation/valid/annotations_valid.csv', usecols=['filename', 'class'])

merged_df = pd.concat([df1, df2, df3])
merged_df = merged_df.drop_duplicates(subset=['filename'])
merged_df.to_csv('labels.csv', index = False)

# Juntar as imagens
folder1 = 'Constellation/train'
folder2 = 'Constellation/test'
folder3 = 'Constellation/valid'

destination_folder = 'images'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def copy_images(source_folder):
    for filename in os.listdir(source_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            source_path = os.path.join(source_folder, filename)
            destination_path = os.path.join(destination_folder, filename)
            shutil.copy(source_path, destination_path)
            print(f"Imagem {filename} copiada.")

copy_images(folder1)
copy_images(folder2)
copy_images(folder3)

print("Juntando as imagens concluído.")

# remover as duplicatas
seen_files = set()

for filename in os.listdir(destination_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        if filename in seen_files:
            os.remove(os.path.join(destination_folder, filename))
            print(f"Imagem duplicada {filename} removida.")
        else:
            seen_files.add(filename)

print("Remoção de imagens duplicadas concluída.")

time.sleep(5)

# Verificar se cada imagem tem uma label no csv
csv_file = 'labels.csv'
df_csv = pd.read_csv(csv_file)

image_files = os.listdir(destination_folder)

for filename in image_files:
    if filename not in df_csv['filename'].values:
        print(f"A imagem {filename} não tem uma classe associada no CSV.")

print("Verificação concluída.")

# remover as imagens sem label

df_csv = pd.read_csv(csv_file)
image_files = os.listdir(destination_folder)
image_files_with_class = []

for filename in image_files:
    if filename in df_csv['filename'].values:
        image_files_with_class.append(filename)
    else:
        os.remove(os.path.join(destination_folder, filename))
        print(f"A imagem {filename} foi removida por não ter uma classe associada no CSV.")

print("Remoção de imagens sem classe concluída.")
