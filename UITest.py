import cv2
import numpy

def luminancia(imagem):
    # Verificar se a imagem é colorida
    if len(imagem.shape) < 3:
        return imagem

    # Aplicar a fórmula de luminância
    luminancia = numpy.dot(imagem[..., :3], [0.299, 0.587, 0.114])
    return luminancia.astype(numpy.uint8)

def limiarizacao(imagem, limiar):
    binaria = luminancia(imagem)
    for x in range(binaria.shape[0]):
        for y in range(binaria.shape[1]):
            if binaria[x, y] > limiar:
                binaria[x, y] = 255
            else:
                binaria[x, y] = 0
                
    cv2.imwrite("imagemLimiarizada.png", binaria)
    
def niveisDeLimiarizacao(imagem, limiares, niveis):
    img_multilevel = numpy.zeros_like(imagem)
    
    for i in range(len(niveis)):
        if i == 0:
            img_multilevel[imagem < limiares[i]] = niveis[i]
        else:
            img_multilevel[(imagem >= limiares[i-1]) & (imagem < limiares[i])] = niveis[i]
    
    return img_multilevel


def multilimiarizacao(imagem, limiares, niveis):
    binaria = niveisDeLimiarizacao(luminancia(imagem), limiares, niveis)
    
    cv2.imwrite("imagemMultiLimiarizada.png", binaria)

# Carregar a imagem
imagem = cv2.imread('trabalho.png', cv2.IMREAD_COLOR)

# Definir os limiares e os níveis de cinza
limiares = [50, 100, 150]
niveis = [0, 85, 170, 255]  # níveis de cinza para a multilimiarização

