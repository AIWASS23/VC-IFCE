import numpy as np
import matplotlib.pyplot as plt
import cv2

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

    return magnitude_gradiente, angulo_gradiente

def supressao(magnitude_gradiente, angulo_gradiente):
    
    suprimida = np.zeros_like(magnitude_gradiente)
    direcoes = np.rad2deg(angulo_gradiente) % 180

    for i in range(1, magnitude_gradiente.shape[0] - 1):
        for j in range(1, magnitude_gradiente.shape[1] - 1):
            direcao = direcoes[i, j]
            if direcao == 0:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j + 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j - 1])
            elif direcao == 45:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j + 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j - 1])
            elif direcao == 90:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j])
            elif direcao == 135:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j - 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j + 1])

    return suprimida * magnitude_gradiente

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


def canny(imagem, limiar_baixo=50, limiar_alto=150):
    suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
    gradientes, angulos = gradiente(suavizada)
    suprimida = supressao(gradientes, angulos)
    bordas = limiarOtsu(suprimida)  # Usando limiar de Otsu para detectar bordas
    
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
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            # rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diagonal
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + int(diagonal)  # Converter rho para inteiro

            acumulador[rho, t_idx] += 1

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

def houghCirculos(imagem, raio, limiar=100):
    bordas = canny(imagem)
    
    altura, largura = bordas.shape
    acumulador = np.zeros((altura, largura, raio), dtype=np.uint64)
    
    thetas = np.linspace(0, 2*np.pi, 360)
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)
    
    y_idxs, x_idxs = np.nonzero(bordas)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):
            for r in range(raio):
                a = int(x - r * cos_t[theta_idx])
                b = int(y - r * sin_t[theta_idx])

                if 0 <= a < largura and 0 <= b < altura:
                    acumulador[b, a, r] += 1
    
    y_pico, x_pico, r_pico = np.where(acumulador >= limiar)
    result_imagem = imagem.copy()
    
    for i in range(len(x_pico)):
        cv2.circle(result_imagem, (x_pico[i], y_pico[i]), raio, (0, 0, 255), 2)

    cv2.imwrite("circulo.png", result_imagem)
