import cv2
import numpy
import matplotlib.pyplot

def luminancia(imagem):
    # Verificar se a imagem é colorida
    if len(imagem.shape) < 3:
        return imagem

    # Aplicar a fórmula de luminância
    luminancia = numpy.dot(imagem[..., :3], [0.299, 0.587, 0.114])
    return luminancia.astype(numpy.uint8)

def equalizador(imagem):
    altura, largura = imagem.shape[:2]

    # Calculo do Histograma
    frequencia = [0] * 256
    valores_intensidade = [0] * 256
    probabilidade = 0

    for x in range(altura):
        for y in range(largura):  
            frequencia[imagem[x, y]] = frequencia[imagem[x, y]] + 1

    # Calculo de probabilidade e probabilidade acumulada
    pixels = altura * largura
    for i in range(256):
        probabilidade = probabilidade + frequencia[i] / float(pixels)
        valores_intensidade[i] = round(probabilidade * 255)

    # Aplicar equalização ao imagem
    for x in range(altura):
        for y in range(largura): 
            imagem[x, y] = valores_intensidade[imagem[x, y]]

    return imagem, valores_intensidade

def equalizarImagemHistograma(imagem):
    # Carregar a imagem
    imagem = cv2.imread(imagem, 0)
    #imagem = luminancia(imagem)

    # Equalizar o histograma
    imagemEqualizada, histogramaEqualizado = equalizador(imagem)

    # Plotar o histograma equalizado
    matplotlib.pyplot.subplot(121).bar(numpy.arange(256), histogramaEqualizado, color = 'gray')
    matplotlib.pyplot.title('Histograma Equalizado')

    # Mostrar a imagem equalizada
    matplotlib.pyplot.subplot(122).imshow(imagemEqualizada, cmap = "gray")
    matplotlib.pyplot.title('Imagem Equalizada')

    # Salvar o histograma equalizado como imagem PNG
    matplotlib.pyplot.savefig("histograma.png")

    # Salvar a imagem equalizada
    cv2.imwrite("imagemEqualizada.png", imagemEqualizada)

imagemCV = "trabalho.png"
#imagemLumi = cv2.imread("trabalho.png")
#imagemEqualLumi = equalizarImagemHistograma(imagemLumi)
imagemEqualCV = equalizarImagemHistograma(imagemCV)