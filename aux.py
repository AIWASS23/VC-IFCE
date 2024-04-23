# import numpy as np
# import cv2

# # def transformadaDaDistancia(imagem):
# #     distancia = np.zeros_like(imagem, dtype = float)
# #     for i in range(imagem.shape[0]):
# #         for j in range(imagem.shape[1]):
# #             if imagem[i, j]:
# #                 distancia[i, j] = np.min(np.sqrt(
# #                     np.square(i - np.arange(imagem.shape[0]))[:, np.newaxis] + 
# #                     np.square(j - np.arange(imagem.shape[1]))
# #                 ))
# #     return distancia

# # def encontrarMarcadores(imagem):
# #     rotulos = np.zeros_like(imagem, dtype=int)
# #     rotuloAtual = 1
# #     for i in range(imagem.shape[0]):
# #         for j in range(imagem.shape[1]):
# #             if imagem[i, j]:
# #                 if rotulos[i, j] == 0:
# #                     lista = [(i, j)]
# #                     while lista:
# #                         x, y = lista.pop()
# #                         if 0 <= x < imagem.shape[0] and 0 <= y < imagem.shape[1] and imagem[x, y] and rotulos[x, y] == 0:
# #                             rotulos[x, y] = rotuloAtual
# #                             lista.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
# #                     rotuloAtual += 1
# #     return rotulos

# def watershed(imagem, percentual):
    
#     # Suavização da imagem
#     borrado = np.sqrt(
#         np.square(imagem[:-2, :-2] - imagem[2:, 2:]) +
#         np.square(imagem[:-2, 2:] - imagem[2:, :-2])
#     )
    
#     # Binarização
#     corte = np.zeros_like(imagem, dtype=bool)
#     corte[1:-1, 1:-1] = borrado > np.percentile(borrado, percentual)
    
#     # Transformada de distância
#     distancia = np.zeros_like(imagem, dtype = float)
#     for i in range(imagem.shape[0]):
#         for j in range(imagem.shape[1]):
#             if corte[i, j]:
#                 distancia[i, j] = np.min(np.sqrt(
#                     np.square(i - np.arange(imagem.shape[0]))[:, np.newaxis] + 
#                     np.square(j - np.arange(imagem.shape[1]))
#                 ))
    
#     # Marcadores
#     rotulos = np.zeros_like(corte, dtype = int)
#     rotuloAtual = 1
#     for i in range(1, corte.shape[0] - 1):
#         for j in range(1, corte.shape[1] - 1):
#             if corte[i, j]:
#                 if rotulos[i, j] == 0:
#                     lista = [(i, j)]
#                     while lista:
#                         x, y = lista.pop()
#                         if 0 <= x < corte.shape[0] and 0 <= y < corte.shape[1] and corte[x, y] and rotulos[x, y] == 0:
#                             rotulos[x, y] = rotuloAtual
#                             lista.extend([(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)])
#                     rotuloAtual += 1
    
#     rotulos[0, :] = 0
#     rotulos[-1, :] = 0
#     rotulos[:, 0] = 0
#     rotulos[:, -1] = 0
    
#     # Atualização dos marcadores usando a transformada de distância
#     for i in range(distancia.shape[0]):
#         for j in range(distancia.shape[1]):
#             if distancia[i, j] < np.percentile(distancia, 10):
#                 rotulos[i, j] = 0
                
#     # Algoritmo de Watershed simplificado
#     fila = np.zeros_like(rotulos, dtype=bool)
#     fila[1:-1, 1:-1] = 1
    
#     while np.any(fila):
#         x, y = np.unravel_index(np.argmax(distancia * fila), fila.shape)
#         fila[x, y] = 0
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 if rotulos[x + dx, y + dy] == 0 and distancia[x + dx, y + dy] < distancia[x, y]:
#                     rotulos[x + dx, y + dy] = rotulos[x, y]
#                     fila[x + dx, y + dy] = 1
    
#     cores = np.random.randint(0, 255, size=(np.max(rotulos) + 1, 3))
#     imagemColorida = cores[rotulos]
    
#     imagemSegmentada = imagemColorida.astype(np.uint8)
#     cv2.imwrite("watershed.png", imagemSegmentada)

# # Carregar a imagem
# caminho_imagem = "trabalho2.png"
# imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

# # Verificar se a imagem foi carregada corretamente
# if imagem is None:
#     print("Erro ao carregar a imagem.")
# else:
#     # Aplicar a função watershed
#     percentual = 90  # Definir o percentual para a binarização
#     watershed(imagem, percentual)

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.ndimage import label

def watershed(imagem, percentual):
    # Suavização da imagem
    dx = np.gradient(imagem, axis=0)
    dy = np.gradient(imagem, axis=1)
    borrado = np.sqrt(dx**2 + dy**2)
    
    # Binarização
    corte = borrado > np.percentile(borrado, percentual)
    
    # Transformada de distância
    distancia = distance_transform_edt(corte)
    
    # Marcadores
    rotulos, num_rotulos = label(corte)
    rotulos[0, :] = 0
    rotulos[-1, :] = 0
    rotulos[:, 0] = 0
    rotulos[:, -1] = 0
    
    # Atualização dos marcadores usando a transformada de distância
    rotulos[distancia < np.percentile(distancia, 10)] = 0
    
    # Algoritmo de Watershed simplificado
    fila = np.zeros_like(rotulos, dtype=bool)
    fila[1:-1, 1:-1] = 1
    
    while np.any(fila):
        x, y = np.unravel_index(np.argmin(distancia * fila), fila.shape)
        fila[x, y] = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if rotulos[x + dx, y + dy] == 0 and distancia[x + dx, y + dy] < distancia[x, y]:
                    rotulos[x + dx, y + dy] = rotulos[x, y]
                    fila[x + dx, y + dy] = 1
    
    cores = np.random.randint(0, 255, size=(np.max(rotulos) + 1, 3))
    imagemColorida = cores[rotulos]
    
    imagemSegmentada = imagemColorida.astype(np.uint8)
    cv2.imwrite("watershed.png", imagemSegmentada)

# Carregar a imagem
caminho_imagem = "trabalho2.png"
imagem = cv2.imread(caminho_imagem, cv2.IMREAD_GRAYSCALE)

# Verificar se a imagem foi carregada corretamente
if imagem is None:
    print("Erro ao carregar a imagem.")
else:
    # Aplicar a função watershed
    percentual = 90  # Definir o percentual para a binarização
    watershed(imagem, percentual)
