import numpy as np
import cv2

def show_segmentation(mask):
    cv2.imshow('Segmentation Process', mask)
    cv2.waitKey(1)  

def region_growing(imagem, pontosIniciais, intervalo, janela):
    
    if janela % 2 == 0:
        print("A janela deve ser impar 😡!")
        exit()
    
    mascara = np.zeros_like(imagem, dtype = imagem.dtype)
    
    lista = []
    for coordenada in pontosIniciais:
        lista.append(coordenada)
    
    iteration = 0
    while lista:
        iteration += 1
        current_point = lista.pop(0)
        current_value = imagem[current_point[1], current_point[0]]
        mascara[current_point[1], current_point[0]] = 255
        
        if iteration % 10 == 0:
            show_segmentation(mascara)
        
        
        for i in range(-(janela//2), (janela//2) + 1):
            for j in range(-(janela//2), (janela//2) + 1):
                if i == 0 and j == 0:
                    continue
                
                x = current_point[0] + i
                y = current_point[1] + j
                
                if 0 <= x < imagem.shape[1] and 0 <= y < imagem.shape[0]:
                    neighbor_value = imagem[y, x]
                    if np.abs(neighbor_value - current_value) <= intervalo:
                        if mascara[y, x] == 0:
                            lista.append((x, y))
                            mascara[y, x] = 255
    
    return mascara