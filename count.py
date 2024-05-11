# import numpy as np
# import matplotlib.pyplot as plt
# import cv2
# from collections import Counter

# def desfoqueGaussiano(imagem, tamanho_kernel):
#     kernel = np.outer(
#         np.exp(-np.linspace(-1, 1, tamanho_kernel) ** 2),
#         np.exp(-np.linspace(-1, 1, tamanho_kernel) ** 2)
#     )
#     kernel /= kernel.sum()

#     suavizada = cv2.filter2D(imagem, -1, kernel)
#     return suavizada.astype(np.uint8)

# def gradiente(imagem):
#     kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
#     kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

#     gradiente_x = cv2.filter2D(imagem, -1, kernel_x)
#     gradiente_y = cv2.filter2D(imagem, -1, kernel_y)

#     magnitude_gradiente = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
#     angulo_gradiente = np.arctan2(gradiente_y, gradiente_x)

#     return magnitude_gradiente, angulo_gradiente

# def supressao(magnitude_gradiente, angulo_gradiente):
    
#     suprimida = np.zeros_like(magnitude_gradiente)
#     direcoes = np.rad2deg(angulo_gradiente) % 180

#     for i in range(1, magnitude_gradiente.shape[0] - 1):
#         for j in range(1, magnitude_gradiente.shape[1] - 1):
#             direcao = direcoes[i, j]
#             if direcao == 0:
#                 suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j + 1]) and \
#                                   (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j - 1])
#             elif direcao == 45:
#                 suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j + 1]) and \
#                                   (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j - 1])
#             elif direcao == 90:
#                 suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j]) and \
#                                   (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j])
#             elif direcao == 135:
#                 suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j - 1]) and \
#                                   (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j + 1])

#     return suprimida * magnitude_gradiente

# def limiarOtsu(imagem):
#     hist, _ = np.histogram(imagem.flatten(), bins=256, range=[0,256])

#     probabilidade = hist / float(np.sum(hist))

#     melhor_limiar = 0
#     melhor_variancia_intra = 0

#     for t in range(1, 256):
#         w0 = np.sum(probabilidade[:t])
#         w1 = np.sum(probabilidade[t:])

#         u0 = np.sum(np.arange(t) * probabilidade[:t]) / w0 if w0 > 0 else 0
#         u1 = np.sum(np.arange(t, 256) * probabilidade[t:]) / w1 if w1 > 0 else 0

#         variancia_intra = w0 * w1 * ((u0 - u1) ** 2)

#         if variancia_intra > melhor_variancia_intra:
#             melhor_variancia_intra = variancia_intra
#             melhor_limiar = t

#     limiarizado = np.zeros_like(imagem)
#     limiarizado[imagem >= melhor_limiar] = 255

#     return limiarizado


# def canny(imagem, limiar_baixo=50, limiar_alto=150):
#     suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
#     gradientes, angulos = gradiente(suavizada)
#     suprimida = supressao(gradientes, angulos)
#     bordas = limiarOtsu(suprimida)  # Usando limiar de Otsu para detectar bordas
    
#     return suprimida


# def houghLinhas(imagem, limiar=100):
#     bordas = canny(imagem)
    
#     altura, largura = bordas.shape
#     diagonal = np.ceil(np.sqrt(altura ** 2 + largura ** 2))
#     thetas = np.deg2rad(np.arange(-90, 90))
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
#     num_thetas = len(thetas)
    
#     # Calculando a matriz de votação da Transformada de Hough
#     acumulador = np.zeros((2 * int(diagonal), num_thetas), dtype=np.uint64)
#     y_idxs, x_idxs = np.nonzero(bordas)
    
#     for x, y in zip(x_idxs, y_idxs):
#         rhos = np.round(x * cos_t + y * sin_t) + int(diagonal)
#         rhos = rhos.astype(int)  # Converter rho para inteiro
#         rhos_counts = Counter(rhos)
        
#         for rho, count in rhos_counts.items():
#             acumulador[rho, :] += count

#     # Encontrando linhas com votação acima do limiar
#     rhos, thetas = np.where(acumulador >= limiar)
#     result_imagem = imagem.copy()
    
#     for rho, theta in zip(rhos, thetas):
#         a = np.cos(thetas[theta])
#         b = np.sin(thetas[theta])
#         x0 = a * (rho - diagonal)
#         y0 = b * (rho - diagonal)
#         x1 = int(x0 + 1000 * (-b))
#         y1 = int(y0 + 1000 * (a))
#         x2 = int(x0 - 1000 * (-b))
#         y2 = int(y0 - 1000 * (a))
#         cv2.line(result_imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linhas em azul

#     cv2.imwrite("linha.png", result_imagem)

# def houghCirculos(imagem, raio, limiar=100):
#     bordas = canny(imagem)
    
#     altura, largura = bordas.shape
#     acumulador = np.zeros((altura, largura, raio), dtype=np.uint64)
    
#     thetas = np.linspace(0, 2*np.pi, 360)
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
    
#     y_idxs, x_idxs = np.nonzero(bordas)
    
#     for x, y in zip(x_idxs, y_idxs):
#         for theta_idx, theta in enumerate(thetas):
#             for r in range(raio):
#                 a = int(x - r * cos_t[theta_idx])
#                 b = int(y - r * sin_t[theta_idx])

#                 if 0 <= a < largura and 0 <= b < altura:
#                     acumulador[b, a, r] += 1
    
#     y_pico, x_pico, r_pico = np.where(acumulador >= limiar)
#     result_imagem = imagem.copy()
    
#     for i in range(len(x_pico)):
#         cv2.circle(result_imagem, (x_pico[i], y_pico[i]), raio, (0, 0, 255), 2)

#     cv2.imwrite("circulo.png", result_imagem)
    
# def criar_imagem_teste():
#     imagem = np.zeros((200, 200), dtype=np.uint8)

#     # Adicionando algumas linhas
#     cv2.line(imagem, (50, 50), (150, 50), (255,), 2)
#     cv2.line(imagem, (50, 100), (150, 100), (255,), 2)
#     cv2.line(imagem, (50, 150), (150, 150), (255,), 2)

#     # Adicionando alguns círculos
#     cv2.circle(imagem, (100, 50), 30, (255,), 2)
#     cv2.circle(imagem, (100, 100), 30, (255,), 2)
#     cv2.circle(imagem, (100, 150), 30, (255,), 2)

#     return imagem

# imagem_teste = criar_imagem_teste()

# plt.imshow(imagem_teste, cmap='gray')
# plt.title('Imagem de Teste')
# plt.axis('off')
# plt.show()

# # Testar a detecção de linhas
# houghLinhas(imagem_teste)

# # Testar a detecção de círculos
# houghCirculos(imagem_teste, raio=30)

# # Exibir resultados
# imagem_linha = cv2.imread('linha.png', cv2.IMREAD_GRAYSCALE)
# imagem_circulo = cv2.imread('circulo.png', cv2.IMREAD_GRAYSCALE)

# plt.figure(figsize=(10, 5))

# plt.subplot(1, 2, 1)
# plt.imshow(imagem_linha, cmap='gray')
# plt.title('Detecção de Linhas')
# plt.axis('off')

# plt.subplot(1, 2, 2)
# plt.imshow(imagem_circulo, cmap='gray')
# plt.title('Detecção de Círculos')
# plt.axis('off')

# plt.show()

import numpy as np
import matplotlib.pyplot as plt
import cv2
from collections import Counter
import random


def desfoqueGaussiano(imagem, tamanho_kernel):
    kernel = np.outer(
        np.exp(-np.linspace(-1, 1, tamanho_kernel) ** 2),
        np.exp(-np.linspace(-1, 1, tamanho_kernel) ** 2)
    )
    kernel /= kernel.sum()

    suavizada = cv2.filter2D(imagem, -1, kernel)
    return suavizada.astype(np.uint8)

def gradiente(imagem):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradiente_x = cv2.filter2D(imagem, -1, kernel_x)
    gradiente_y = cv2.filter2D(imagem, -1, kernel_y)

    magnitude_gradiente = np.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    angulo_gradiente = np.arctan2(gradiente_y, gradiente_x)
    angulo_gradiente = np.rad2deg(angulo_gradiente)
    
    return magnitude_gradiente, angulo_gradiente

def supressao(magnitude_gradiente, angulo_gradiente):
    suprimida = np.zeros_like(magnitude_gradiente)

    for i in range(1, magnitude_gradiente.shape[0] - 1):
        for j in range(1, magnitude_gradiente.shape[1] - 1):
            angulo = angulo_gradiente[i, j]
            if 0 <= angulo < 22.5 or 157.5 <= angulo <= 180:
                vizinho1 = magnitude_gradiente[i, j + 1]
                vizinho2 = magnitude_gradiente[i, j - 1]
            elif 22.5 <= angulo < 67.5:
                vizinho1 = magnitude_gradiente[i - 1, j + 1]
                vizinho2 = magnitude_gradiente[i + 1, j - 1]
            elif 67.5 <= angulo < 112.5:
                vizinho1 = magnitude_gradiente[i - 1, j]
                vizinho2 = magnitude_gradiente[i + 1, j]
            else:
                vizinho1 = magnitude_gradiente[i - 1, j - 1]
                vizinho2 = magnitude_gradiente[i + 1, j + 1]

            if magnitude_gradiente[i, j] >= vizinho1 and magnitude_gradiente[i, j] >= vizinho2:
                suprimida[i, j] = magnitude_gradiente[i, j]

    return suprimida

def limiarOtsu(imagem):
    hist, _ = np.histogram(imagem.flatten(), bins=256, range=[0,256])

    probabilidade = hist / float(np.sum(hist))

    melhor_limiar = 0
    melhor_variancia_intra = 0

    for t in range(1, 256):
        w0 = np.sum(probabilidade[:t])
        w1 = np.sum(probabilidade[t:])

        u0 = np.sum(np.arange(t) * probabilidade[:t]) / w0 if w0 > 0 else 0
        u1 = np.sum(np.arange(t, 256) * probabilidade[t:]) / w1 if w1 > 0 else 0

        variancia_intra = w0 * w1 * ((u0 - u1) ** 2)

        if variancia_intra > melhor_variancia_intra:
            melhor_variancia_intra = variancia_intra
            melhor_limiar = t

    limiarizado = np.zeros_like(imagem)
    limiarizado[imagem >= melhor_limiar] = 255

    return limiarizado

# def canny(imagem, limiar_baixo=50, limiar_alto=150):
#     suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
#     gradientes, angulos = gradiente(suavizada)
#     suprimida = supressao(gradientes, angulos)
#     bordas = limiarOtsu(suprimida)
    
#     return bordas

def canny(imagem, limiar_baixo=0, limiar_alto=255):
    print("Imagem original:")
    plt.imshow(imagem, cmap='gray')
    plt.title('Imagem Original')
    plt.axis('off')
    plt.show()

    print("Aplicando suavização gaussiana...")
    suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
    plt.imshow(suavizada, cmap='gray')
    plt.title('Suavização Gaussiana')
    plt.axis('off')
    plt.show()

    print("Calculando gradientes...")
    gradientes, angulos = gradiente(suavizada)
    plt.imshow(gradientes, cmap='gray')
    plt.title('Magnitude do Gradiente')
    plt.axis('off')
    plt.show()

    print("Suprimindo não-máximos...")
    suprimida = supressao(gradientes, angulos)
    plt.imshow(suprimida, cmap='gray')
    plt.title('Supressão de Não-Máximos')
    plt.axis('off')
    plt.show()

    print("Aplicando limiar de Otsu para detectar bordas...")
    bordas = limiarOtsu(suprimida)
    plt.imshow(bordas, cmap='gray')
    plt.title('Detecção de Bordas (Canny)')
    plt.axis('off')
    plt.show()
    
    return bordas


def houghLinhas(imagem, limiar=100):
    bordas = canny(imagem)
    
    altura, largura = bordas.shape
    diagonal = np.ceil(np.sqrt(altura ** 2 + largura ** 2))
    thetas = np.deg2rad(np.arange(-90, 90))
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    num_thetas = len(thetas)
    
    # Calculando a matriz de votação da Transformada de Hough
    acumulador = np.zeros((2 * int(diagonal), num_thetas), dtype=np.uint64)
    y_idxs, x_idxs = np.nonzero(bordas)
    
    for x, y in zip(x_idxs, y_idxs):
        rhos = np.round(x * cos_t + y * sin_t) + int(diagonal)
        rhos = rhos.astype(int)  # Converter rho para inteiro
        rhos_counts = Counter(rhos)
        
        for rho, count in rhos_counts.items():
            acumulador[rho, :] += count

    # Encontrando linhas com votação acima do limiar
    rhos, thetas = np.where(acumulador >= limiar)
    result_imagem = imagem.copy()
    
    for rho, theta in zip(rhos, thetas):
        a = np.cos(thetas[theta])
        b = np.sin(thetas[theta])
        x0 = a * (rho - diagonal)
        y0 = b * (rho - diagonal)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result_imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linhas em azul

    cv2.imwrite("linha.png", result_imagem)
    
def AHT(imagem, raio_desejado, limiar):
    bordas = canny(imagem)

    acumulador = np.zeros(imagem.shape, dtype=np.uint64)
    gradientes_x, gradientes_y = np.gradient(imagem)
    gradientes_mag = np.sqrt(gradientes_x**2 + gradientes_y**2)

    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            if bordas[i, j] > 0:
                for r in range(1, min(i, imagem.shape[0]-i, j, imagem.shape[1]-j)):
                    acumulador[i - r, j] += gradientes_mag[i, j]
                    acumulador[i + r, j] += gradientes_mag[i, j]
                    acumulador[i, j - r] += gradientes_mag[i, j]
                    acumulador[i, j + r] += gradientes_mag[i, j]
    
    picos = np.where(acumulador >= limiar)
    centros_circulos = [(picos[0][i], picos[1][i]) for i in range(len(picos[0]))]
    
    result_imagem = cv2.cvtColor(imagem, cv2.COLOR_GRAY2BGR)
    for centro in centros_circulos:
        cor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv2.circle(result_imagem, centro, raio_desejado, cor, 2)

    return cv2.imwrite("circulo.png", result_imagem)

    
# def houghCirculos(imagem, raio, limiar=100):
#     bordas = canny(imagem)
    
#     altura, largura = bordas.shape
#     acumulador = np.zeros((altura, largura, raio), dtype=np.uint64)
    
#     thetas = np.linspace(0, 2*np.pi, 360)
#     cos_t = np.cos(thetas)
#     sin_t = np.sin(thetas)
    
#     y_idxs, x_idxs = np.nonzero(bordas)
    
#     for x, y in zip(x_idxs, y_idxs):
#         for theta_idx, theta in enumerate(thetas):
#             for r in range(raio):
#                 a = int(x - r * cos_t[theta_idx])
#                 b = int(y - r * sin_t[theta_idx])

#                 if 0 <= a < largura and 0 <= b < altura:
#                     acumulador[b, a, r] += 1
    
#     y_pico, x_pico, r_pico = np.where(acumulador >= limiar)
#     result_imagem = imagem.copy()
    
#     for i in range(len(x_pico)):
#         cv2.circle(result_imagem, (x_pico[i], y_pico[i]), raio, (0, 0, 255), 2)

#     cv2.imwrite("circulo.png", result_imagem)
    
imagem_teste = cv2.imread("trabalho2.png", cv2.IMREAD_GRAYSCALE)

# Testar a detecção de linhas
#houghLinhas(imagem_teste)

# Testar a detecção de círculos
AHT(imagem_teste, 200, 255)

# def houghCirculos(imagem, raio, limiar):
#     bordas = canny(imagem)
    
#     min_raio = raio - 10
#     max_raio = raio + 10
    
#     altura, largura = bordas.shape
#     acumulador = numpy.zeros((altura, largura, max_raio - min_raio + 1 ), dtype = bordas.dtype) # dtype = numpy.uint64
    
#     y_idxs, x_idxs = numpy.nonzero(bordas)
#     for i in range(len(x_idxs)):
#         x = x_idxs[i]
#         y = y_idxs[i]
        
#         # Iterar sobre os valores de raio
#         for r in range(min_raio, max_raio + 1):
#             # Iterar sobre os valores de ângulo
#             for theta_idx in range(360):
#                 theta = numpy.deg2rad(theta_idx)
                
#                 # Calcular o centro do círculo
#                 a = int(x - r * numpy.cos(theta))
#                 b = int(y - r * numpy.sin(theta))
                
#                 # Verificar se o centro está dentro da imagem
#                 if 0 <= a < largura and 0 <= b < altura:
#                     # Acumular voto no bin correspondente
#                     acumulador[b, a, r - min_raio] += 1
    
#     # Encontrar os picos no acumulador
#     y_pico, x_pico, r_pico = numpy.where(acumulador >= limiar)
#     result_imagem = imagem.copy()
    
#     # Desenhar os círculos detectados
#     for i in range(len(x_pico)):
#         cor = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
#         cv2.circle(result_imagem, (x_pico[i], y_pico[i]), r_pico[i] + min_raio, cor, 2)

#     cv2.imwrite("circulo.png", result_imagem)
