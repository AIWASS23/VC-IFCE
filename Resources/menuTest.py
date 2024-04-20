import cv2
import numpy
import argparse

def gaussiana(x, y, sigma):
    return (1 / (2 * numpy.pi * sigma ** 2)) * numpy.exp(-((x ** 2) + (y ** 2)) / (2 * sigma ** 2))

def convolucao(imagem, kernel):
    m, n = imagem.shape
    k_m, k_n = kernel.shape
    imagem_filtrada = numpy.zeros((m - k_m + 1, n - k_n + 1))

    for i in range(m - k_m + 1):
        for j in range(n - k_n + 1):
            imagem_filtrada[i, j] = numpy.sum(imagem[i:i + k_m, j:j + k_n] * kernel)

    return imagem_filtrada

def kernel_gaussiano(tamanho_janela, sigma):
  
    if tamanho_janela % 2 == 0:
        raise ValueError("Tamanho da janela deve ser ímpar")

    centro = (tamanho_janela // 2, tamanho_janela // 2)
    kernel = numpy.zeros((tamanho_janela, tamanho_janela))

    for i in range(tamanho_janela):
        for j in range(tamanho_janela):
            x = i - centro[0]
            y = j - centro[1]
            kernel[i, j] = gaussiana(x, y, sigma)

    return kernel

def filtroGaussiano(imagem, tamanho_janela, sigma):
    kernel = kernel_gaussiano(tamanho_janela, sigma)
    imagemGaussiana = convolucao(imagem, kernel)

    cv2.imwrite("gaussiana.png", imagemGaussiana)

def filtroMediana(imagem, tamanho_janela):
    m, n = imagem.shape
    imagemMediana = numpy.zeros([m, n])

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
    
def filtroMedia(imagem, tamanho_janela):
    m, n = imagem.shape
    imagemMedia = numpy.zeros([m, n])

    for i in range(tamanho_janela // 2, m - tamanho_janela // 2):
        for j in range(tamanho_janela // 2, n - tamanho_janela // 2):
            janela = [
                imagem[i - k][j - l]
                for k in range(tamanho_janela)
                for l in range(tamanho_janela)
            ]
            soma = sum(janela)
            imagemMedia[i, j] = soma / (tamanho_janela ** 2)

    imagemMedia = imagemMedia.astype(numpy.uint8)
    cv2.imwrite("media.png", imagemMedia)
    
def int_validator(value):
    try:
        int_value = int(value)

        if int_value < 1:
            raise ValueError

        return int_value
    except ValueError:
        if isinstance(value, str):
            try:
                float_value = float(value)
                if float_value.is_integer() and float_value > 0:
                    return int(float_value)
            except ValueError:
                pass
        raise argparse.ArgumentTypeError(f"{value} não é um número inteiro positivo")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--tamanho_janela", type = int, default = 3)
    parser.add_argument("-f", "--tipo_filtro", type = str, default = "mediana")
    parser.add_argument("entrada", type = str)
    args = parser.parse_args()

    imagem = cv2.imread(args.entrada, 0)
    if args.tipo_filtro == "mediana":
        filtroMediana(imagem, args.tamanho_janela)
    elif args.tipo_filtro == "media":
        filtroMedia(imagem, args.tamanho_janela)
    else:
        print("Tipo de filtro inválido!")