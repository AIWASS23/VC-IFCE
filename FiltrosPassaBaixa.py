import cv2
import numpy
import argparse

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
    imagemGaussiana = numpy.zeros((m - k_m + 1, n - k_n + 1))

    for i in range(m - k_m + 1):
        for j in range(n - k_n + 1):
            if tamanho_janela % 2 == 0:
                imagemGaussiana[i, j] = numpy.sum(imagem[i:i + k_m, j:j + k_n] * (kernel * 2.0))
            else:
                imagemGaussiana[i, j] = numpy.sum(imagem[i:i + k_m, j:j + k_n] * kernel)

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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-w", "--tamanho_janela", type = int, default = 3)
    parser.add_argument("-f", "--tipo_filtro", type = str, default = "mediana")
    parser.add_argument("-s", "--sigma", type = float, default = 1.0 )
    parser.add_argument("entrada", type = str)
    args = parser.parse_args()

    imagem = cv2.imread(args.entrada, 0)
    if args.tipo_filtro == "mediana":
        filtroMediana(imagem, args.tamanho_janela)
    elif args.tipo_filtro == "media":
        filtroMedia(imagem, args.tamanho_janela)
    elif args.tipo_filtro == "gaussiana":
        filtroGaussiano(imagem, args.tamanho_janela, args.sigma)
    else:
        print("Tipo de filtro inválido!")