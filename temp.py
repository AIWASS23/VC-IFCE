import cv2
import numpy
import matplotlib.pyplot

def filtroMedia(imagem, tamanho_janela):
    m, n = imagem.shape  # O(1)
    
    # Espaço: O(m * n)
    imagemMedia = numpy.zeros([m, n], dtype = imagem.dtype)  # O(m * n)

    # Temporal: O(m * n * (tamanho_janela^2 + m + 5))
    for i in range(tamanho_janela // 2, m - tamanho_janela // 2):  # O(m)
        for j in range(tamanho_janela // 2, n - tamanho_janela // 2):  # O(n)
            # Espaço: O(tamanho_janela^2)
            janela = [
                imagem[i - k][j - l]
                for k in range(tamanho_janela)  # O(tamanho_janela)
                for l in range(tamanho_janela)  # O(tamanho_janela)
            ]
            soma = sum(janela)  # O(tamanho_janela^2)
            imagemMedia[i, j] = soma / (tamanho_janela ** 2)  # O(1)

    # Espaço: O(m * n)
    imagemMedia = imagemMedia.astype(numpy.uint8)  # O(m * n)
    
    # Temporal: O(m * n)
    cv2.imwrite("media.png", imagemMedia)  # O(m * n)

# Complexidade Temporal: O(m * n * (tamanho_janela^2 + m + 5))
# Complexidade Espacial: O(m * n) + O(tamanho_janela^2)
    
def filtroMediana(imagem, tamanho_janela):
    m, n = imagem.shape
    imagemMediana = numpy.zeros([m, n], dtype = imagem.dtype)

    for i in range(tamanho_janela // 2, m - tamanho_janela // 2):
        for j in range(tamanho_janela // 2, n - tamanho_janela // 2):
            janela = [
                imagem[i - k][j - l]
                for k in range(tamanho_janela)
                for l in range(tamanho_janela)
            ]
            temp = sorted(janela)
            imagemMediana[i, j] = temp[len(temp) // 2]

    imagemMediana = imagemMediana.astype(numpy.uint8)
    cv2.imwrite("mediana.png", imagemMediana)
    
def filtroGaussiano(imagem, tamanho_janela, sigma):
  
    if tamanho_janela % 2 == 0:
        centro = (tamanho_janela // 2, tamanho_janela // 2)
        kernel = numpy.zeros((tamanho_janela + 1, tamanho_janela + 1))

        for i in range(tamanho_janela + 1):
            for j in range(tamanho_janela + 1):
                x = i - centro[0]
                y = j - centro[1]
                kernel[i, j] = (1 / (2 * numpy.pi * sigma ** 2)) * numpy.exp(-((x ** 2) + (y ** 2)) / (2 * sigma ** 2))

        kernel /= 2.0
    
    else: 
        centro = (tamanho_janela // 2, tamanho_janela // 2)
        kernel = numpy.zeros((tamanho_janela, tamanho_janela))

        for i in range(tamanho_janela):
            for j in range(tamanho_janela):
                x = i - centro[0]
                y = j - centro[1]
                kernel[i, j] = (1 / (2 * numpy.pi * sigma ** 2)) * numpy.exp(-((x ** 2) + (y ** 2)) / (2 * sigma ** 2))

    m, n = imagem.shape
    k_m, k_n = kernel.shape
    imagemGaussiana = numpy.zeros((m - k_m + 1, n - k_n + 1), dtype = imagem.dtype)

    for i in range(m - k_m + 1):
        for j in range(n - k_n + 1):
            if tamanho_janela % 2 == 0:
                imagemGaussiana[i, j] = numpy.sum(imagem[i:i + k_m, j:j + k_n] * (kernel * 2.0))
            else:
                imagemGaussiana[i, j] = numpy.sum(imagem[i:i + k_m, j:j + k_n] * kernel)

    cv2.imwrite("gaussiana.png", imagemGaussiana)

def filtroLaplaciano(imagem, mascara, tamanho_janela):
    
    if len(mascara) != tamanho_janela or len(mascara[0]) != tamanho_janela: 
        raise ValueError("Tamanho da máscara inválido para o filtro Laplaciano (deve ser igual ao tamanho da janela).")
        
    m, n = imagem.shape

    imagemLaplaciana = numpy.zeros(
        (m - tamanho_janela + 1, n - tamanho_janela + 1), 
        dtype = imagem.dtype
    )

    for i in range(m - tamanho_janela + 1):
        for j in range(n - tamanho_janela + 1):
            janela = imagem[i : i + tamanho_janela, j : j + tamanho_janela]
            valor_laplaciano = numpy.sum(mascara * janela)
            imagemLaplaciana[i, j] = valor_laplaciano

    imagemLaplaciana = imagemLaplaciana.astype(numpy.uint8)
    cv2.imwrite("laplaciana.png", imagemLaplaciana)
        
def filtroPrewitt(imagem):
    
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
    imagemPrewitt = (imagemPrewitt - numpy.min(imagemPrewitt)) / (numpy.max(imagemPrewitt) - numpy.min(imagemPrewitt)) * 255
    imagemPrewitt = imagemPrewitt.astype(numpy.uint8)
    cv2.imwrite("prewitt.png", imagemPrewitt)
    
def filtroSobel(imagem):

    mascara_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    mascara_y = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    m, n = imagem.shape

    imagemSobelX = numpy.zeros((m - 2, n - 2), dtype = imagem.dtype)
    imagemSobelY = numpy.zeros((m - 2, n - 2), dtype = imagem.dtype)

    for i in range(m - 2):
        for j in range(n - 2):
            janela = imagem[i : i + 3, j : j + 3]
            valor_sobel_x = numpy.sum(mascara_x * janela)
            valor_sobel_y = numpy.sum(mascara_y * janela)
            imagemSobelX[i, j] = valor_sobel_x
            imagemSobelY[i, j] = valor_sobel_y

    imagemSobel = numpy.sqrt(imagemSobelX ** 2 + imagemSobelY ** 2)
    imagemSobel = (imagemSobel - numpy.min(imagemSobel)) / (numpy.max(imagemSobel) - numpy.min(imagemSobel)) * 255
    imagemSobel = imagemSobel.astype(numpy.uint8)
    cv2.imwrite("sobel.png", imagemSobel)

def calcularHistograma(imagem, canal):
    eixo_y = [0] * 256

    if len(imagem.shape) == 3:
        for x in range(imagem.shape[0]):
            for y in range(imagem.shape[1]):
                intensidade = imagem[x, y, canal]
                eixo_y[intensidade] += 1
    elif len(imagem.shape) == 2:
        for x in range(imagem.shape[0]):
            for y in range(imagem.shape[1]):
                intensidade = imagem[x, y]
                eixo_y[intensidade] += 1

    return eixo_y

def verificarImagem(imagem):
    if len(imagem.shape) == 3:
        if imagem.shape[2] == 3:
            return "RGB"
        elif imagem.shape[2] == 4:
            return "RGBA"
    elif len(imagem.shape) == 2:
        return "Grayscale"
    else:
        raise ValueError("Formato de imagem não suportado.")
    
def apresentarHistograma(imagem):
    formato = verificarImagem(imagem)

    if formato == "RGB":
        eixoVermelho = calcularHistograma(imagem, 0)
        eixoVerde = calcularHistograma(imagem, 1)
        eixoAzul = calcularHistograma(imagem, 2)
        matplotlib.pyplot.subplot(131).bar(numpy.arange(256), eixoVermelho, color = 'red')
        matplotlib.pyplot.subplot(132).bar(numpy.arange(256), eixoVerde, color = 'green')
        matplotlib.pyplot.subplot(133).bar(numpy.arange(256), eixoAzul, color = 'blue')
        matplotlib.pyplot.title('Histograma RGB')
        matplotlib.pyplot.savefig("histogramaRGB.png")

    elif formato == "RGBA":
        eixoVermelho = calcularHistograma(imagem, 0)
        eixoVerde = calcularHistograma(imagem, 1)
        eixoAzul = calcularHistograma(imagem, 2)
        matplotlib.pyplot.subplot(131).bar(numpy.arange(256), eixoVermelho, color = 'red')
        matplotlib.pyplot.subplot(132).bar(numpy.arange(256), eixoVerde, color = 'green')
        matplotlib.pyplot.subplot(133).bar(numpy.arange(256), eixoAzul, color = 'blue')
        matplotlib.pyplot.title('Histograma RGBA')
        matplotlib.pyplot.savefig("histogramaRGBA.png")

    elif formato == "Grayscale":
        eixoCinza = calcularHistograma(imagem, 0)
        matplotlib.pyplot.subplot(111).bar(numpy.arange(256), eixoCinza, color = 'gray')
        matplotlib.pyplot.title('Histograma Grayscale')
        matplotlib.pyplot.savefig("histogramaGrayscale.png")

def equalizador(imagem):
    altura, largura = imagem.shape[:2]

    frequencia = [0] * 256
    valores_intensidade = [0] * 256
    probabilidade = 0

    for x in range(altura):
        for y in range(largura):  
            frequencia[imagem[x, y]] = frequencia[imagem[x, y]] + 1

    pixels = altura * largura
    for i in range(256):
        probabilidade = probabilidade + frequencia[i] / float(pixels)
        valores_intensidade[i] = round(probabilidade * 255)

    for x in range(altura):
        for y in range(largura): 
            imagem[x, y] = valores_intensidade[imagem[x, y]]

    return imagem, valores_intensidade

def equalizarImagemHistograma(imagem):
    imagem = luminancia(imagem)
    imagemEqualizada, histogramaEqualizado = equalizador(imagem)

    matplotlib.pyplot.subplot(121).bar(numpy.arange(256), histogramaEqualizado, color = 'gray')
    matplotlib.pyplot.title('Histograma Equalizado')

    matplotlib.pyplot.subplot(122).imshow(imagemEqualizada, cmap = "gray")
    matplotlib.pyplot.title('Imagem Equalizada')

    matplotlib.pyplot.savefig("histograma.png")
    cv2.imwrite("imagemEqualizada.png", imagemEqualizada)
    
def luminancia(imagem):
    if len(imagem.shape) < 3:
        return imagem

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
    if len(niveis) > len(limiares) + 1:
        print("Erro: O número de níveis (-n) não deve ser 2 a mais que o número de limiares (-m).")
        exit()
    
    imagemMultilevel = numpy.zeros_like(imagem)
    
    for i in range(len(niveis)):
        if i == 0:
            imagemMultilevel[imagem < limiares[i]] = niveis[i]
        elif i == len(niveis) - 1:
            imagemMultilevel[imagem >= limiares[i-1]] = niveis[i]
        else:
            imagemMultilevel[(imagem >= limiares[i-1]) & (imagem < limiares[i])] = niveis[i]
    
    return imagemMultilevel

def multilimiarizacao(imagem, limiares, niveis):
    binaria = niveisDeLimiarizacao(luminancia(imagem), limiares, niveis)
    cv2.imwrite("imagemMultiLimiarizada.png", binaria)
