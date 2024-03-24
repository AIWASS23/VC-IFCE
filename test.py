import cv2
import numpy

def filtroPrewitt(imagem):
    
    # Máscaras de Prewitt
    mascara_x = numpy.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])
    mascara_y = numpy.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])

    m, n = imagem.shape

    imagemPrewittX = numpy.zeros((m - 2, n - 2), dtype = imagem.dtype)
    imagemPrewittY = numpy.zeros((m - 2, n - 2), dtype = imagem.dtype)

    for i in range(m - 2):
        for j in range(n - 2):
            janela = imagem[i : i + 3, j : j + 3]
            valor_prewitt_x = numpy.sum(mascara_x * janela)
            valor_prewitt_y = numpy.sum(mascara_y * janela)
            imagemPrewittX[i, j] = valor_prewitt_x
            imagemPrewittY[i, j] = valor_prewitt_y

    imagemPrewitt = numpy.sqrt(imagemPrewittX ** 2 + imagemPrewittY ** 2)

    imagemPrewitt = (imagemPrewitt / numpy.max(imagemPrewitt)) * 255
    #imagemPrewitt = (imagemPrewitt - numpy.min(imagemPrewitt)) / (numpy.max(imagemPrewitt) - numpy.min(imagemPrewitt)) * 255

    imagemPrewitt = imagemPrewitt.astype(numpy.uint8)
    cv2.imwrite("prewitt.png", imagemPrewitt)

# Carregue uma imagem em escala de cinza
imagemCeara = cv2.imread('trabalho.png', cv2.IMREAD_GRAYSCALE)
imagemFortaleza = cv2.imread('trabalho.png')

imagemVovo = cv2.imwrite("vovo.png", imagemCeara)
imagemLeao = cv2.imwrite("leao.png", imagemFortaleza)

# Aplique o filtro de Prewitt
imagemPrewitt = filtroPrewitt(imagemCeara)
#imagemPrewitt = filtroPrewitt(imagemFortaleza)

