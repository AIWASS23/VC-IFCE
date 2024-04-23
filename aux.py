# import numpy as np
# import cv2

# def crescimentoDeRegiao(imagem, coordenadas, janela, limite):
#     linhas, colunas = imagem.shape
#     visitado = np.zeros_like(imagem)
#     regiao = np.zeros_like(imagem)
#     lista = [coordenadas]
#     intensidade_semente = np.int16(imagem[coordenadas])
    
#     while lista:
#         x, y = lista.pop()
#         intensidade_atual = np.int16(imagem[x, y])
#         if visitado[x, y] == 0 and abs(intensidade_atual - intensidade_semente) < limite:
#             regiao[x, y] = 255
#             visitado[x, y] = 1
#             for dx in range(-janela, janela + 1):
#                 for dy in range(-janela, janela + 1):
#                     x_novo, y_novo = np.clip(x + dx, 0, linhas-1), np.clip(y + dy, 0, colunas-1)
#                     if visitado[x_novo, y_novo] == 0:
#                         lista.append((x_novo, y_novo))

#     imagemRegiao = regiao.astype(np.uint8)
#     cv2.imwrite("regiao.png", imagemRegiao)

# imagem = cv2.imread('trabalho2.png', cv2.IMREAD_GRAYSCALE)

# # Escolha um ponto de semente (coordenadas)
# coordenadas = (50, 50)

# # Defina o tamanho da janela (vizinhança)
# janela = 3

# # Defina o limite para o crescimento da região
# limite = 10

# # Aplique o crescimento da região
# regiao = crescimentoDeRegiao(imagem, coordenadas, janela, limite)

import numpy as np
import cv2
from collections import deque

def crescimentoDeRegiao(imagem, coordenadas, janela, limite):
    linhas, colunas = imagem.shape
    visitado = np.zeros_like(imagem, dtype = bool)
    regiao = np.zeros_like(imagem, dtype = imagem.dtype)
    fila = deque([coordenadas])
    intensidade_semente = imagem[coordenadas].astype(imagem.dtype)
    
    while fila:
        x, y = fila.popleft()
        intensidade_atual = imagem[x, y].astype(imagem.dtype)
        
        if not visitado[x, y] and abs(int(intensidade_atual) - int(intensidade_semente)) < limite:
            regiao[x, y] = 255
            visitado[x, y] = True
            
            for dx in range(-janela, janela + 1):
                for dy in range(-janela, janela + 1):
                    x_novo, y_novo = x + dx, y + dy
                    if 0 <= x_novo < linhas and 0 <= y_novo < colunas and not visitado[x_novo, y_novo]:
                        fila.append((x_novo, y_novo))

    imagemRegiao = regiao.astype(np.uint8)
    cv2.imwrite("regiao.png", imagemRegiao)
    

# Exemplo de uso
imagem = cv2.imread('trabalho2.png', cv2.IMREAD_GRAYSCALE)
coordenadas = (500, 50)
janela = 1
limite = 200
regiao = crescimentoDeRegiao(imagem, coordenadas, janela, limite)

