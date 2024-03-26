import matplotlib.pyplot as plt
import numpy as np
import cv2

def calcular_histograma_canal(imagem, canal):
    eixo_y = [0] * 256

    if len(imagem.shape) == 3:
        for x in range(imagem.shape[0]):
            for y in range(imagem.shape[1]):
                intensidade = imagem[x, y, canal]
                eixo_y[intensidade] += 1
    elif len(imagem.shape) == 2:
        for x in range(imagem.shape[0]):
            for y in range(imagem.shape[1]):
                intensidade = imagem[x, y]
                eixo_y[intensidade] += 1

    return eixo_y

def verificar_formato_imagem(imagem):
    if len(imagem.shape) == 3:
        if imagem.shape[2] == 3:
            return "RGB"
        elif imagem.shape[2] == 4:
            return "RGBA"
    elif len(imagem.shape) == 2:
        return "Grayscale"
    else:
        raise ValueError("Formato de imagem não suportado.")
    
def calcular_histograma_manual(imagem):
    formato = verificar_formato_imagem(imagem)

    if formato == "RGB":
        eixo_y_r = calcular_histograma_canal(imagem, 0)
        eixo_y_g = calcular_histograma_canal(imagem, 1)
        eixo_y_b = calcular_histograma_canal(imagem, 2)
        return eixo_y_r, eixo_y_g, eixo_y_b

    elif formato == "RGBA":
        eixo_y_r = calcular_histograma_canal(imagem, 0)
        eixo_y_g = calcular_histograma_canal(imagem, 1)
        eixo_y_b = calcular_histograma_canal(imagem, 2)
        return eixo_y_r, eixo_y_g, eixo_y_b

    elif formato == "Grayscale":
        eixo_y = calcular_histograma_canal(imagem, 0)
        print("Tons de Cinza")
        print(eixo_y)
        return eixo_y

#imagem = cv2.imread("trabalho.png", cv2.IMREAD_COLOR) # RBG ✅
imagem = cv2.imread("trabalho.png", cv2.IMREAD_GRAYSCALE) # tons de cinza ✅
#imagem = cv2.imread("trabalho.png", cv2.IMREAD_UNCHANGED) # RBGA ✅

formato = verificar_formato_imagem(imagem)

if formato == "RGB":
    eixo_y_r, eixo_y_g, eixo_y_b = calcular_histograma_manual(imagem)

elif formato == "RGBA":
    eixo_y_r, eixo_y_g, eixo_y_b = calcular_histograma_manual(imagem)

elif formato == "Grayscale":
    eixo_y = calcular_histograma_manual(imagem)