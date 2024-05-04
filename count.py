import numpy as np
import cv2

def houghLinhas(imagem, limiar=100):
    bordas = canny(imagem)
    
    altura, largura = bordas.shape
    diagonal = np.ceil(np.sqrt(altura ** 2 + largura **2 ))
    rhos = np.linspace(-diagonal, diagonal, int(diagonal * 2))
    thetas = np.deg2rad(np.arange(-90, 90))
    
    acumulador = np.zeros((len(rhos), len(thetas)), dtype=imagem.dtype)
    indiceY, indiceX = np.nonzero(bordas)
    
    for i in range(len(indiceX)):
        x = indiceX[i]
        y = indiceY[i]
        
        for j in range(len(thetas)):
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + int(diagonal)
            acumulador[rho, j] += 1
    
    rhos_idx, thetas_idx = np.where(acumulador > limiar)
    result_imagem = imagem.copy()
    
    for i in range(len(rhos_idx)):
        rho = rhos[rhos_idx[i]]
        theta = thetas[thetas_idx[i]]
        
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        
        cv2.line(result_imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linhas em azul

    return result_imagem

def houghCirculos(imagem, raio, limiar=100):
    bordas = canny(imagem)
    
    acumulador = np.zeros_like(imagem, dtype=np.uint64)
    indiceY, indiceX = np.nonzero(bordas)
    
    for i in range(len(indiceX)):
        x = indiceX[i]
        y = indiceY[i]
        
        for theta in np.linspace(0, 2*np.pi, 360):
            a = int(x - raio * np.cos(theta))
            b = int(y - raio * np.sin(theta))
            
            if 0 <= a < acumulador.shape[1] and 0 <= b < acumulador.shape[0]:
                acumulador[b, a] += 1
    
    y_pico, x_pico = np.where(acumulador >= limiar)
    result_imagem = imagem.copy()
    
    for i in range(len(x_pico)):
        cv2.circle(result_imagem, (x_pico[i], y_pico[i]), raio, (0, 0, 255), 2)  # Círculos em vermelho

    return result_imagem

def canny(imagem, limiar_baixo=50, limiar_alto=150):
    suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
    magnitude_gradiente, angulo_gradiente = gradiente(suavizada)
    suprimida = supressao(magnitude_gradiente, angulo_gradiente)
    bordas = limiarOtsu(suprimida)
    
    return bordas

def desfoqueGaussiano(imagem, tamanho_kernel):
    kernel = np.outer(np.exp(-np.linspace(-1, 1, tamanho_kernel)**2), np.exp(-np.linspace(-1, 1, tamanho_kernel)**2))
    kernel /= kernel.sum()
    
    suavizada = cv2.filter2D(imagem, -1, kernel)
    
    return suavizada.astype(np.uint8)

def gradiente(imagem):
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    gradiente_x = cv2.filter2D(imagem, -1, kernel_x)
    gradiente_y = cv2.filter2D(imagem, -1, kernel_y)
    
    magnitude_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    angulo_gradiente = np.arctan2(gradiente_y, gradiente_x)
    
    return magnitude_gradiente, angulo_gradiente

def supressao(magnitude_gradiente, angulo_gradiente):
    suprimida = np.zeros_like(magnitude_gradiente)
    
    for i in range(1, magnitude_gradiente.shape[0]-1):
        for j in range(1, magnitude_gradiente.shape[1]-1):
            angulo = angulo_gradiente[i, j]
            if -np.pi/8 <= angulo < np.pi/8 or 7*np.pi/8 <= angulo or -7*np.pi/8 > angulo:
                if magnitude_gradiente[i, j] > magnitude_gradiente[i, j+1] and magnitude_gradiente[i, j] > magnitude_gradiente[i, j-1]:
                    suprimida[i, j] = magnitude_gradiente[i, j]
            elif np.pi/8 <= angulo < 3*np.pi/8 or -7*np.pi/8 <= angulo < -5*np.pi/8:
                if magnitude_gradiente[i, j] > magnitude_gradiente[i-1, j+1] and magnitude_gradiente[i, j] > magnitude_gradiente[i+1, j-1]:
                    suprimida[i, j] = magnitude_gradiente[i, j]
            elif 3*np.pi/8 <= angulo < 5*np.pi/8 or -5*np.pi/8 <= angulo < -3*np.pi/8:
                if magnitude_gradiente[i, j] > magnitude_gradiente[i-1, j] and magnitude_gradiente[i, j] > magnitude_gradiente[i+1, j]:
                    suprimida[i, j] = magnitude_gradiente[i, j]
            else:
                if magnitude_gradiente[i, j] > magnitude_gradiente[i-1, j-1] and magnitude_gradiente[i, j] > magnitude_gradiente[i+1, j+1]:
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

imagem = cv2.imread('01.jpg', 0)  
# imagem_linhas = houghLinhas(imagem)
imagem_circulos = houghCirculos(imagem, 20)

cv2.imwrite('bola.png', imagem_circulos)
# cv2.imwrite('linha.png', imagem_linhas)
