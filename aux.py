import numpy as np
import matplotlib.pyplot as plt

def houghLinhas(imagem, limiar=100):
    bordas = canny(imagem)
    
    # Definir os limites do espaço de Hough
    altura, largura = bordas.shape
    diagonal = np.ceil(np.sqrt(altura ** 2 + largura **2 ))
    rhos = np.linspace(-diagonal, diagonal, int(diagonal * 2))
    thetas = np.deg2rad(np.arange(-90, 90))
    
    # Acumulador de Hough
    acumulador = np.zeros((len(rhos), len(thetas)), dtype = imagem.dtype)
    
    # Encontrar índices dos pixels de borda
    indiceY, indiceX = np.nonzero(bordas)
    
    # Votar no acumulador
    for i in range(len(indiceX)):
        x = indiceX[i]
        y = indiceY[i]
        
        for j in range(len(thetas)):
            rho = int(x * np.cos(thetas[j]) + y * np.sin(thetas[j])) + int(diagonal)
            acumulador[rho, j] += 1
    
    # Encontrar os picos no acumulador
    rhos_idx, thetas_idx = np.where(acumulador > limiar)
    
    # Desenhar as linhas detectadas
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
        
        plt.plot([x1, x2], [y1, y2], 'r')
    
    plt.imshow(imagem, cmap = 'gray')
    plt.title('Detecção de Linhas usando Transformada de Hough')
    plt.show()
    

def houghCirculos(imagem, raio, limiar=100):
    # Detecção de bordas usando o algoritmo Canny
    bordas = canny(imagem)
    
    # Acumulador de Hough para círculos
    acumulador = np.zeros_like(imagem, dtype=np.uint64)
    i, indiceX = np.nonzero(bordas)
    
    # Votar no acumulador
    for i in range(len(indiceX)):
        x = indiceX[i]
        y = i[i]
        
        for theta in np.linspace(0, 2*np.pi, 360):
            a = int(x - raio * np.cos(theta))
            b = int(y - raio * np.sin(theta))
            
            if 0 <= a < acumulador.shape[1] and 0 <= b < acumulador.shape[0]:
                acumulador[b, a] += 1
    
    # Encontrar os picos no acumulador
    y_pico, x_pico = np.where(acumulador > limiar)
    
    # Desenhar os círculos detectados
    plt.imshow(imagem, cmap='gray')
    for i in range(len(x_pico)):
        plt.scatter(x_pico[i], y_pico[i], s=100, c='r', marker='o')
    plt.title('Detecção de Círculos usando Transformada de Hough')
    plt.show()

def canny(imagem, limiar_baixo=50, limiar_alto=150):
    # Aplicar o algoritmo Canny para detecção de bordas
    suavizada = desfoqueGaussiano(imagem, tamanho_kernel = 5)
    magnitude_gradiente, angulo_gradiente = gradiente(suavizada)
    suprimida = supressao(magnitude_gradiente, angulo_gradiente)
    bordas = limiarDuplo(suprimida, limiar_baixo, limiar_alto)
    
    return bordas

def desfoqueGaussiano(imagem, tamanho_kernel):
    # Implementação básica de desfoque gaussiano
    kernel = np.outer(np.exp(-np.linspace(-1, 1, tamanho_kernel)**2), np.exp(-np.linspace(-1, 1, tamanho_kernel)**2))
    kernel /= kernel.sum()
    
    suavizada = np.zeros_like(imagem, dtype=np.float64)
    
    imagem_padded = np.pad(imagem, ((tamanho_kernel//2, tamanho_kernel//2), (tamanho_kernel//2, tamanho_kernel//2)), mode='constant')
    
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            suavizada[i, j] = np.sum(kernel * imagem_padded[i:i+tamanho_kernel, j:j+tamanho_kernel])
    
    return suavizada.astype(np.uint8)

def gradiente(imagem):
    # Cálculo do gradiente usando operadores Sobel
    kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    
    gradiente_x = convolucao(imagem, kernel_x)
    gradiente_y = convolucao(imagem, kernel_y)
    
    magnitude_gradiente = np.sqrt(gradiente_x**2 + gradiente_y**2)
    angulo_gradiente = np.arctan2(gradiente_y, gradiente_x)
    
    return magnitude_gradiente, angulo_gradiente

def convolucao(imagem, kernel):
    # Convolução 2D
    kernel = np.flipud(np.fliplr(kernel))
    saida = np.zeros_like(imagem)
    
    imagem_padded = np.pad(imagem, ((kernel.shape[0]//2, kernel.shape[0]//2), (kernel.shape[1]//2, kernel.shape[1]//2)), mode='constant')
    
    for i in range(imagem.shape[0]):
        for j in range(imagem.shape[1]):
            saida[i, j] = np.sum(imagem_padded[i:i+kernel.shape[0], j:j+kernel.shape[1]] * kernel)
    
    return saida

def supressao(magnitude_gradiente, angulo_gradiente):
    # Supressão de não-máximo
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

def limiarDuplo(imagem, limiar_baixo, limiar_alto):
    # Limiar duplo para detecção de bordas
    bordas_fortes = imagem > limiar_alto
    bordas_fracas = (imagem >= limiar_baixo) & (imagem <= limiar_alto)
    
    # Conectar componentes fracos aos fortes
    for i in range(1, imagem.shape[0]-1):
        for j in range(1, imagem.shape[1]-1):
            if bordas_fracas[i, j]:
                if bordas_fortes[i-1:i+2, j-1:j+2].any():
                    bordas_fortes[i, j] = True
    
    return bordas_fortes.astype(np.uint8) * 255

# Testando a função de detecção de linhas
caminho_imagem = '01.jpg'  # Substitua pelo caminho para a sua imagem
imagem = plt.imread(caminho_imagem)
imagem_cinza = np.mean(imagem, axis=2).astype(np.uint8)
houghLinhas(imagem_cinza)

# Testando a função de detecção de círculos
caminho_imagem = '01.jpg'  # Substitua pelo caminho para a sua imagem
imagem = plt.imread(caminho_imagem)
imagem_cinza = np.mean(imagem, axis=2).astype(np.uint8)
houghCirculos(imagem_cinza, raio=50)
