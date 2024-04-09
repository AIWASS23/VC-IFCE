import cv2
import numpy as np
from matplotlib import pyplot as plt

def translacao(imagem, tx, ty, height, width):
    rows, cols = imagem.shape
    img_trans = np.zeros((height, width), dtype = np.uint8)
    
    for i in range(rows):
        for j in range(cols):
            if 0 <= i + ty < height and 0 <= j + tx < width:
                img_trans[i + ty, j + tx] = imagem[i, j]
                
    return img_trans

def rotacao(img, angle, height, width):
    
    rows, cols = img.shape
    img_rot = np.zeros((height, width), dtype=np.uint8)
    center_x, center_y = cols / 2, rows / 2
    
    for i in range(rows):
        for j in range(cols):
            x = int((j - center_x) * np.cos(np.radians(angle)) - (i - center_y) * np.sin(np.radians(angle)) + center_x)
            y = int((j - center_x) * np.sin(np.radians(angle)) + (i - center_y) * np.cos(np.radians(angle)) + center_y)
            
            if 0 <= y < rows and 0 <= x < cols:
                img_rot[i, j] = img[y, x]
                
    return img_rot

def escala(img, fx, fy):
    
    rows, cols = img.shape
    height, width = int(rows * fy), int(cols * fx)
    img_scaled = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(height):
        for j in range(width):
            x = int(j / fx)
            y = int(i / fy)
            
            if 0 <= y < rows and 0 <= x < cols:
                img_scaled[i, j] = img[y, x]
                
    return img_scaled

# Carregando a imagem
src = cv2.imread('trabalho.png', 0)

# Aplicando as transformações utilizando as funções criadas
trans = translacao(src, 100, 50, 288, 288)
#rot = rotacao(src, 90, 288, 288)
esc = escala(src, 2, 2)

# Mostrando as imagens
plt.subplot(131), plt.imshow(trans, cmap='gray'), plt.title('Translação')
#plt.subplot(132), plt.imshow(rot, cmap='gray'), plt.title('Rotação')
plt.subplot(133), plt.imshow(esc, cmap='gray'), plt.title('Escala')

plt.show()
