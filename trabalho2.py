import cv2
import numpy as np

def erosion(image, kernel_size):
    height, width = image.shape
    kh, kw = kernel_size
    
    # Inicializar imagem de saída com zeros
    output = np.zeros((height, width), dtype=np.uint8)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            minimum = 255  # valor máximo para imagens em escala de cinza
            for m in range(kh):
                for n in range(kw):
                    if kernel[m, n] == 1:
                        minimum = min(minimum, image[i-1+m, j-1+n])
            output[i, j] = minimum
    
    return output

def dilation(image, kernel_size):
    height, width = image.shape
    kh, kw = kernel_size
    
    # Inicializar imagem de saída com zeros
    output = np.zeros((height, width), dtype = image.dtype)
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            maximum = 0  # valor mínimo para imagens em escala de cinza
            for m in range(kh):
                for n in range(kw):
                    if kernel[m, n] == 1:
                        maximum = max(maximum, image[i-1+m, j-1+n])
            output[i, j] = maximum
    
    return output

# Carregar a imagem usando OpenCV
image = cv2.imread('exemplo.jpg', cv2.IMREAD_GRAYSCALE)

# Escolher o tamanho do kernel (por exemplo, 3x3 ou 5x5)
kernel_size = (5, 5)

# Criar o elemento estruturante (kernel)
kernel = np.ones(kernel_size, dtype=np.uint8)

# Aplicar erosão
eroded_image = erosion(image, kernel_size)
cv2.imshow('Erosion', eroded_image)
