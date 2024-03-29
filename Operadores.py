import matplotlib.pyplot
import numpy
import cv2
import argparse

# Questão 7

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

# Questão 8

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
    imagem = luminancia(imagem)

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
    
# Questão 9

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
    
# Questão 10

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

if __name__ == "__main__":
    
    # Configuração do CLI
    parser = argparse.ArgumentParser(
        description = "Processamento de imagens utilizando diferentes operações."
    )
    parser.add_argument(
        "-o", 
        "--operacao",
        choices = ['histograma', 'equalizacao', 'limiarizacao', 'multilimiarizacao'],
        type = str, 
        default = "histograma", 
        help = "Opções de Operação: histograma, equalizacao, limiarizacao e multilimiarizacao."
    )
    parser.add_argument(
        "-l", 
        "--limiar", 
        type = int, 
        default = 127, 
        help = "Valor do Limiar para Limiarização."
    )
    parser.add_argument(
        "-i", 
        "--imagem", 
        required = True,
        help = "Caminho da imagem de entrada."
    )
    parser.add_argument(
        "-m", 
        "--limiares", 
        nargs = '+', 
        type = int, 
        default = [50, 100, 150],
        help = "Lista de limiares para a multilimiarização."
    )
    parser.add_argument(
        "-n", 
        "--niveis", 
        nargs='+', 
        type=int, 
        default = [0, 85, 170, 255],
        help = "Lista de níveis de cinza para a multilimiarização."
    )
    args = parser.parse_args()
    
    # Leitura da imagem
    imagem = cv2.imread(args.imagem)

    # Operações
    if args.operacao == "histograma":
        apresentarHistograma(imagem) 
    elif args.operacao == "equalizacao":
        equalizarImagemHistograma(imagem)
    elif args.operacao == "limiarizacao":
        limiarizacao(imagem, args.limiar)
    elif args.operacao == "multilimiarizacao":
        multilimiarizacao(imagem, args.limiares, args.niveis)
    else:
        print("Tipo de operacão inválido!")
