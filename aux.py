import numpy as np
import cv2
import collections

def limiarizacao(imagem, limiar, valor_maximo):
    imagem_limiarizada = np.where(imagem > limiar, valor_maximo, 0).astype(np.uint8)
    return imagem_limiarizada

def crescimento_de_regiao(imagem, semente, limite):
    altura, largura = imagem.shape
    regiao = np.zeros_like(imagem, dtype=np.uint8)
    
    sementes = [semente]
    
    while sementes:
        x, y = sementes.pop()
        
        if x > 0 and np.abs(int(imagem[x, y]) - int(imagem[x - 1, y])) < limite and regiao[x - 1, y] == 0:
            regiao[x - 1, y] = 255
            sementes.append((x - 1, y))
        
        if x < altura - 1 and np.abs(int(imagem[x, y]) - int(imagem[x + 1, y])) < limite and regiao[x + 1, y] == 0:
            regiao[x + 1, y] = 255
            sementes.append((x + 1, y))
        
        if y > 0 and np.abs(int(imagem[x, y]) - int(imagem[x, y - 1])) < limite and regiao[x, y - 1] == 0:
            regiao[x, y - 1] = 255
            sementes.append((x, y - 1))
        
        if y < largura - 1 and np.abs(int(imagem[x, y]) - int(imagem[x, y + 1])) < limite and regiao[x, y + 1] == 0:
            regiao[x, y + 1] = 255
            sementes.append((x, y + 1))
    
    return regiao

def rotular_regioes(imagem):
    altura, largura = imagem.shape
    rotulada = np.zeros_like(imagem, dtype=np.int32)
    contador_rotulos = 1
    
    for i in range(altura):
        for j in range(largura):
            if imagem[i, j] == 255 and rotulada[i, j] == 0:
                regiao = crescimento_de_regiao(imagem, (i, j), 10)
                rotulada[regiao == 255] = contador_rotulos
                contador_rotulos += 1
                
    return rotulada

def mapeamento_cores(num_rotulos):
    np.random.seed(42)  
    return np.random.randint(0, 256, size=(num_rotulos, 3), dtype=np.uint8)

def convolucao(imagem, kernel):
    linhas, colunas = imagem.shape
    klinhas, kcolunas = kernel.shape
    altura_pad = klinhas // 2
    largura_pad = kcolunas // 2
    imagem_padding = np.pad(imagem, ((altura_pad, altura_pad), (largura_pad, largura_pad)), mode='constant')
    saida = np.zeros_like(imagem)
    
    for i in range(linhas):
        for j in range(colunas):
            saida[i, j] = np.sum(imagem_padding[i:i+klinhas, j:j+kcolunas] * kernel)
    
    return saida

def erosao(imagem, kernel):
    altura, largura = imagem.shape
    
    kaltura, klargura = kernel.shape
    
    altura_pad = kaltura // 2
    largura_pad = klargura // 2
    
    saida = np.zeros((altura, largura), dtype=imagem.dtype)
    
    for i in range(altura_pad, altura - altura_pad):
        for j in range(largura_pad, largura - largura_pad):
            minimum = 255
            for m in range(kaltura):
                for n in range(klargura):
                    if kernel[m, n] == 1:
                        minimum = min(minimum, imagem[i - altura_pad + m, j - largura_pad + n])
            saida[i, j] = minimum
            
    return saida.astype(np.uint8)

def dilatacao(imagem, kernel):
    altura, largura = imagem.shape
    
    kaltura, klargura = kernel.shape
    
    altura_pad = kaltura // 2
    largura_pad = klargura // 2
    
    saida = np.zeros((altura, largura), dtype=imagem.dtype)
    
    for i in range(altura_pad, altura - altura_pad):
        for j in range(largura_pad, largura - largura_pad):
            maximum = 0
            for m in range(kaltura):
                for n in range(klargura):
                    if kernel[m, n] == 1:
                        maximum = max(maximum, imagem[i - altura_pad + m, j - largura_pad + n])
            saida[i, j] = maximum
            
    return saida.astype(np.uint8)

def aplicar_abertura_manual(imagem, kernel):
    imagem_erodida = erosao(imagem, kernel)
    imagem_abertura = dilatacao(imagem_erodida, kernel)
    return imagem_abertura

def watershed(imagem, limiar, maximo, kernel):
    
    kernel = np.ones((kernel, kernel), np.uint8)    
    cinza_filtrado = aplicar_abertura_manual(imagem, kernel)
    
    sobelx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = np.abs(convolucao(cinza_filtrado, sobelx))
    grad_y = np.abs(convolucao(cinza_filtrado, sobely))
    
    magnitude_gradiente = np.sqrt(grad_x**2 + grad_y**2)
    
    thresholded = limiarizacao(cinza_filtrado, limiar, maximo)
    
    marcadores = rotular_regioes(thresholded)
    
    marcadores[magnitude_gradiente == 0] = 0
    
    fila = collections.deque()
    rotulado = np.zeros_like(marcadores)
    rotulo = 1
    
    for i in range(marcadores.shape[0]):
        for j in range(marcadores.shape[1]):
            if marcadores[i, j] == 1 and rotulado[i, j] == 0:
                rotulado[i, j] = rotulo
                fila.append((i, j))
                
                while fila:
                    x, y = fila.popleft()
                    for nx, ny in zip([x-1, x+1, x, x], [y, y, y-1, y+1]):
                        if 0 <= nx < marcadores.shape[0] and 0 <= ny < marcadores.shape[1]:
                            if marcadores[nx, ny] == 1 and rotulado[nx, ny] == 0:
                                rotulado[nx, ny] = rotulo
                                fila.append((nx, ny))
                
                rotulo += 1
    
    rotulos_unicos = np.unique(rotulado)
    mapeamento_colorido = mapeamento_cores(len(rotulos_unicos))
    
    rotulado_colorido = np.zeros((rotulado.shape[0], rotulado.shape[1], 3), dtype = np.uint8)
    
    for rotulo in rotulos_unicos:
        if rotulo == 0:
            continue
        rotulado_colorido[rotulado == rotulo] = mapeamento_colorido[rotulo]
    
    cv2.imwrite("watershed.png", rotulado_colorido)

imagem = cv2.imread("trabalho2.png")
watershed(imagem)
