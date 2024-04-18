import cv2
import numpy
import argparse

# Questão 1

def filtragemFrequencia(imagem, tipo_filtro, tamanho, tamanho_banda):
   
    imagem_fft = numpy.fft.fft2(imagem)
    imagem_fft_shift = numpy.fft.fftshift(imagem_fft)
    
    linhas, colunas = imagem.shape
    centroLinha = linhas // 2
    centroColuna = colunas // 2
    
    filtro = numpy.zeros((linhas, colunas), dtype = imagem.dtype)
    
    if tipo_filtro == 'passa-baixa':
        filtro[centroLinha - tamanho : centroLinha + tamanho, centroColuna - tamanho : centroColuna + tamanho] = 1
        
    elif tipo_filtro == 'passa-alta':
        filtro[centroLinha - tamanho : centroLinha + tamanho, centroColuna - tamanho : centroColuna + tamanho] = 1
        filtro = numpy.ones((linhas, colunas), numpy.uint8) - filtro
        
    elif tipo_filtro == 'passa-banda':
        filtro[centroLinha - tamanho : centroLinha + tamanho, centroColuna - tamanho - tamanho_banda : centroColuna + tamanho_banda] = 1
        filtro = numpy.ones((linhas, colunas), numpy.uint8) - filtro
        
    elif tipo_filtro == 'rejeita-banda':
        filtro[centroLinha - tamanho : centroLinha + tamanho, centroColuna - tamanho - tamanho_banda : centroColuna + tamanho_banda] = 0
        filtro = numpy.ones((linhas, colunas), numpy.uint8) - filtro
    
    imagem_filtrada_fft_shift = imagem_fft_shift * filtro
    
    imagem_filtrada_fft = numpy.fft.ifftshift(imagem_filtrada_fft_shift)
    imagem_filtrada = numpy.fft.ifft2(imagem_filtrada_fft)
    imagem_filtrada = numpy.abs(imagem_filtrada)
    
    imagem_filtrada = (255 * (imagem_filtrada - numpy.min(imagem_filtrada)) / (numpy.max(imagem_filtrada) - numpy.min(imagem_filtrada))).astype(numpy.uint8)
    cv2.imwrite("filtragemFrequencia.png", imagem_filtrada)
    
# Questão 2

def erosao(imagem, kernel):
    altura, largura = imagem.shape
    
    saida = numpy.zeros((altura, largura), dtype = imagem.dtype)
    
    for i in range(1, altura - 1):
        for j in range(1, largura - 1):
            minimum = 255 
            for m in range(kernel):
                for n in range(kernel):
                    if kernel[m, n] == 1:
                        minimum = min(minimum, imagem[i - 1 + m, j - 1 + n])
            saida[i, j] = minimum
            
    imagemErosada = saida.astype(numpy.uint8)
    cv2.imwrite("erosao.png", imagemErosada)

def dilatacao(imagem, kernel):
    altura, largura = imagem.shape
    
    saida = numpy.zeros((altura, largura), dtype = imagem.dtype)
    
    for i in range(1, altura-1):
        for j in range(1, largura-1):
            maximum = 0 
            for m in range(kernel):
                for n in range(kernel):
                    if kernel[m, n] == 1:
                        maximum = max(maximum, imagem[i - 1 + m, j - 1 + n])
            saida[i, j] = maximum
    
    imagemDilatada = saida.astype(numpy.uint8)
    cv2.imwrite("dilatada.png", imagemDilatada)
    
# Questão 3

def limiarizacaoMediaMovel(imagem, tamanho_janela):

    altura, largura = imagem.shape
    saida = numpy.zeros((altura, largura), dtype = imagem.dtype)
    
    # Calcular a média móvel para cada pixel
    for i in range(altura):
        for j in range(largura):
            # Definir os limites da janela
            inicioAltura = max(0, i - tamanho_janela // 2)
            fimAltura = min(altura, i + tamanho_janela // 2)
            inicioLargura = max(0, j - tamanho_janela // 2)
            fimLargura = min(largura, j + tamanho_janela // 2)
            
            # Calcular a média móvel
            media = numpy.mean(imagem[inicioAltura : fimAltura, inicioLargura : fimLargura])
            
            # Limiarização
            if imagem[i, j] > media:
                saida[i, j] = 255
            else:
                saida[i, j] = 0
    
    imagemLimiarizada = saida.astype(numpy.uint8)
    cv2.imwrite("limiarizacaoPorMedia.png", imagemLimiarizada)

if __name__ == "__main__":
    
# --------------- configuração do CLI ------------------------- #  
  
    parser = argparse.ArgumentParser(
        description = "Processamento de imagens utilizando diferentes transformadas."
    )
    parser.add_argument(
        "-k", 
        "--tamanho_kernel", 
        type = int, 
        default = 3,
        help = "Valor do kernel que será convertido para (w x w)."
    )
    
    parser.add_argument(
        "-b", 
        "--tamanho_banda", 
        type = int, 
        default = 3,
        help = "Largura da banda do filtro em pixels para os tipos de filtro passa-banda e rejeita-banda. Define a largura da faixa ativa ou inativa no filtro. Deve ser um valor inteiro."
    )
    
    parser.add_argument(
        "-t", 
        "--transformacao", 
        type = str, 
        default = "frequencia",
        help = "Transformacão que será aplicada na imagem"
    )
    parser.add_argument(
        "-f", 
        "--tipo_filtro", 
        type = str, 
        default = "passa baixa",
        help = "Opções de Filtragem: passa baixa, passa alta, passa banda e rejeita banda"
    )
    parser.add_argument(
        "-r", 
        "--raio", 
        type = int, 
        default = 3, 
        required = False,
        help = "Tamanho do filtro em pixels. Especifica o raio da região ativa no filtro passa-baixa, passa-alta, passa-banda ou rejeita-banda. Deve ser um valor inteiro."

    )
    parser.add_argument(
        "-i", 
        "--imagem", 
        required = True,
        help = "Caminho da imagem de entrada."
    )
    args = parser.parse_args()

# ------------------- Leituras Adicionais ---------------------- #

    if args.tamanho_kernel <= 0:
        parser.error(f"O valor de {args.tamanho_janela} para janela é inválido. Deve ser um número inteiro positivo.")

    imagem = cv2.imread(args.imagem, 0)

    if args.transformacao == "frequencia":
        filtragemFrequencia(imagem, args.tipo_filtro, args.raio, args.tamanho_banda)
    elif args.transformacao == "erosao":
        erosao(imagem, args.tamanho_kernel)
    elif args.transformacao == "dilatacao":
        dilatacao(imagem, args.tamanho_kernel)
    elif args.transformacao == "movel":
        limiarizacaoMediaMovel(imagem, args.tamanho_kernel)
    else:
        print("Tipo de filtro inválido!")