import cv2
import numpy
import argparse

# ------------------------ Filtros Passa Baixa ----------------------------- #

# Questão 1

def filtroMedia(imagem, tamanho_janela):
    m, n = imagem.shape
    imagemMedia = numpy.zeros([m, n], dtype = imagem.dtype)

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
    
# Questão 2

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
    
# Questão 3

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
    
#------------------------ Filtros Passa Alta -----------------------------------#

# Questão 4

def filtroLaplaciano(imagem, mascara, tamanho_janela):
    
    if len(mascara) != tamanho_janela or len(mascara[0]) != tamanho_janela: 
        print("Tamanho da máscara inválido para o filtro Laplaciano (deve ser igual ao tamanho da janela).")
        exit()
        
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
    
# Questão 5
    
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
    
    #imagemPrewitt = (imagemPrewitt / numpy.max(imagemPrewitt)) * 255
    imagemPrewitt = (imagemPrewitt - numpy.min(imagemPrewitt)) / (numpy.max(imagemPrewitt) - numpy.min(imagemPrewitt)) * 255

    imagemPrewitt = imagemPrewitt.astype(numpy.uint8)
    cv2.imwrite("prewitt.png", imagemPrewitt)
    
# Questão 6

def filtroSobel(imagem):

    # Máscaras de Sobel
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

    # imagemSobel = (imagemSobel / numpy.max(imagemSobel)) * 255
    imagemSobel = (imagemSobel - numpy.min(imagemSobel)) / (numpy.max(imagemSobel) - numpy.min(imagemSobel)) * 255

    imagemSobel = imagemSobel.astype(numpy.uint8)
    cv2.imwrite("sobel.png", imagemSobel)


# -------------------------------- Menu ------------------------------------------ #
    
if __name__ == "__main__":
    
# --------------- configuração do CLI ------------------------- #  
  
    parser = argparse.ArgumentParser(
        description = "Processamento de imagens utilizando diferentes filtros."
    )
    parser.add_argument(
        "-w", 
        "--tamanho_janela", 
        type = int, 
        default = 3,
        help = "Valor do tamanho da janela que será convertido para (w x w)."
    )
    parser.add_argument(
        "-m", 
        "--mascara", 
        type = str, 
        required = False,
        help = "Mascará no formato .txt utilizado exclusivamente no filtro lapraciano."
    )
    parser.add_argument(
        "-f", 
        "--tipo_filtro", 
        type = str, 
        default = "mediana",
        help = "Opções de Filtros: media, mediana, gaussiano, laplaciano, prewitt e sobel."
    )
    parser.add_argument(
        "-s", 
        "--sigma", 
        type = float, 
        default = 1.0, 
        required = False,
        help = "Valor do sigma utilizado exclusivamente no filtro gaussiano."
    )
    parser.add_argument(
        "-i", 
        "--imagem", 
        required = True,
        help = "Caminho da imagem de entrada."
    )
    args = parser.parse_args()

# ------------------- Leituras Adicionais ---------------------- #

    if args.tamanho_janela <= 0:
        parser.error(f"O valor de {args.tamanho_janela} para janela é inválido. Deve ser um número inteiro positivo.")

    imagem = cv2.imread(args.imagem, 0) # trabalha com 2 canais

# ------------------- Filtros Passa Baixa --------------------- #

    if args.tipo_filtro == "media":
        filtroMedia(imagem, args.tamanho_janela)
    elif args.tipo_filtro == "mediana":
        filtroMediana(imagem, args.tamanho_janela)
    elif args.tipo_filtro == "gaussiano":
        filtroGaussiano(imagem, args.tamanho_janela, args.sigma)
    
# ------------------- Filtros Passa Alta ----------------------- #
    
    elif args.tipo_filtro == "laplaciano":
        mascara = numpy.loadtxt(args.mascara, dtype = numpy.float32)
        filtroLaplaciano(imagem, mascara, args.tamanho_janela)
    elif args.tipo_filtro == "prewitt":
        filtroPrewitt(imagem)
    elif args.tipo_filtro == "sobel":
        filtroSobel(imagem)
    else:
        print("Tipo de filtro inválido!")