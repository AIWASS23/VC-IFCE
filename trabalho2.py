import cv2
import numpy
import argparse
import collections

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
    
# Questão 4

def crescimentoDeRegiao(imagem, coordenadasX, coordenadasY, janela, limite):
    
    if coordenadasX < 0 or coordenadasX >= imagem.shape[0] or coordenadasY < 0 or coordenadasY >= imagem.shape[1]:
        print("Coordenadas deve está na imagem 😡")
        exit()

    linhas, colunas = imagem.shape
    visitado = numpy.zeros_like(imagem, dtype = bool)
    regiao = numpy.zeros_like(imagem, dtype = imagem.dtype)
    fila = collections.deque([(coordenadasX, coordenadasY)])
    intensidade_semente = imagem[coordenadasX, coordenadasY].astype(imagem.dtype)
    
    while fila:
        x, y = fila.popleft()
        intensidade_atual = imagem[x, y].astype(imagem.dtype)
        
        if not visitado[x, y] and abs(int(intensidade_atual) - int(intensidade_semente)) < limite:
            regiao[x, y] = 255
            visitado[x, y] = True
            
            for dx in range(-janela, janela + 1):
                for dy in range(-janela, janela + 1):
                    x_novo, y_novo = x + dx, y + dy
                    if 0 <= x_novo < linhas and 0 <= y_novo < colunas and not visitado[x_novo, y_novo]:
                        fila.append((x_novo, y_novo))

    imagemRegiao = regiao.astype(numpy.uint8)
    cv2.imwrite("regiao.png", imagemRegiao)
    
# Questão 5

def limiarizacao(imagem, limiar):
    imagem_limiarizada = numpy.where(imagem > limiar, 255, 0).astype(numpy.uint8)
    return imagem_limiarizada

def crescimento_de_regiao(imagem, semente, limite):
    altura, largura = imagem.shape
    regiao = numpy.zeros_like(imagem, dtype=numpy.uint8)
    
    sementes = [semente]
    
    while sementes:
        x, y = sementes.pop()
        
        if x > 0 and numpy.abs(int(imagem[x, y]) - int(imagem[x - 1, y])) < limite and regiao[x - 1, y] == 0:
            regiao[x - 1, y] = 255
            sementes.append((x - 1, y))
        
        if x < altura - 1 and numpy.abs(int(imagem[x, y]) - int(imagem[x + 1, y])) < limite and regiao[x + 1, y] == 0:
            regiao[x + 1, y] = 255
            sementes.append((x + 1, y))
        
        if y > 0 and numpy.abs(int(imagem[x, y]) - int(imagem[x, y - 1])) < limite and regiao[x, y - 1] == 0:
            regiao[x, y - 1] = 255
            sementes.append((x, y - 1))
        
        if y < largura - 1 and numpy.abs(int(imagem[x, y]) - int(imagem[x, y + 1])) < limite and regiao[x, y + 1] == 0:
            regiao[x, y + 1] = 255
            sementes.append((x, y + 1))
    
    return regiao

def rotular_regioes(imagem):
    altura, largura = imagem.shape
    rotulada = numpy.zeros_like(imagem, dtype=numpy.int32)
    contador_rotulos = 1
    
    for i in range(altura):
        for j in range(largura):
            if imagem[i, j] == 255 and rotulada[i, j] == 0:
                regiao = crescimento_de_regiao(imagem, (i, j), 10)
                rotulada[regiao == 255] = contador_rotulos
                contador_rotulos += 1
                
    return rotulada

def mapeamento_cores(num_rotulos):
    numpy.random.seed(42)  
    return numpy.random.randint(0, 256, size=(num_rotulos, 3), dtype=numpy.uint8)

def convolucao(imagem, kernel):
    linhas, colunas = imagem.shape
    klinhas, kcolunas = kernel.shape
    altura_pad = klinhas // 2
    largura_pad = kcolunas // 2
    imagem_padding = numpy.pad(imagem, ((altura_pad, altura_pad), (largura_pad, largura_pad)), mode='constant')
    saida = numpy.zeros_like(imagem)
    
    for i in range(linhas):
        for j in range(colunas):
            saida[i, j] = numpy.sum(imagem_padding[i:i+klinhas, j:j+kcolunas] * kernel)
    
    return saida

def erosao(imagem, kernel):
    altura, largura = imagem.shape
    
    kaltura, klargura = kernel.shape
    
    altura_pad = kaltura // 2
    largura_pad = klargura // 2
    
    saida = numpy.zeros((altura, largura), dtype=imagem.dtype)
    
    for i in range(altura_pad, altura - altura_pad):
        for j in range(largura_pad, largura - largura_pad):
            minimum = 255
            for m in range(kaltura):
                for n in range(klargura):
                    if kernel[m, n] == 1:
                        minimum = min(minimum, imagem[i - altura_pad + m, j - largura_pad + n])
            saida[i, j] = minimum
            
    return saida.astype(numpy.uint8)

def dilatacao(imagem, kernel):
    altura, largura = imagem.shape
    
    kaltura, klargura = kernel.shape
    
    altura_pad = kaltura // 2
    largura_pad = klargura // 2
    
    saida = numpy.zeros((altura, largura), dtype=imagem.dtype)
    
    for i in range(altura_pad, altura - altura_pad):
        for j in range(largura_pad, largura - largura_pad):
            maximum = 0
            for m in range(kaltura):
                for n in range(klargura):
                    if kernel[m, n] == 1:
                        maximum = max(maximum, imagem[i - altura_pad + m, j - largura_pad + n])
            saida[i, j] = maximum
            
    return saida.astype(numpy.uint8)

def aplicar_abertura_manual(imagem, kernel):
    imagem_erodida = erosao(imagem, kernel)
    imagem_abertura = dilatacao(imagem_erodida, kernel)
    return imagem_abertura

def watershed(imagem, limiar, kernel):
    
    kernel = numpy.ones((kernel, kernel), numpy.uint8)    
    cinza_filtrado = aplicar_abertura_manual(imagem, kernel)
    
    sobelx = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobely = numpy.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    
    grad_x = numpy.abs(convolucao(cinza_filtrado, sobelx))
    grad_y = numpy.abs(convolucao(cinza_filtrado, sobely))
    
    magnitude_gradiente = numpy.sqrt(grad_x**2 + grad_y**2)
    
    thresholded = limiarizacao(cinza_filtrado, limiar)
    
    marcadores = rotular_regioes(thresholded)
    
    marcadores[magnitude_gradiente == 0] = 0
    
    fila = collections.deque()
    rotulado = numpy.zeros_like(marcadores)
    rotulo = 1
    
    for i in range(marcadores.shape[0]):
        for j in range(marcadores.shape[1]):
            if marcadores[i, j] == 1 and rotulado[i, j] == 0:
                rotulado[i, j] = rotulo
                fila.append((i, j))
                
                while fila:
                    x, y = fila.popleft()
                    for nx, ny in zip([x-1, x+1, x, x], [y, y, y-1, y+1]):
                        if 0 <= nx < marcadores.shape[0] and 0 <= ny < marcadores.shape[1]:
                            if marcadores[nx, ny] == 1 and rotulado[nx, ny] == 0:
                                rotulado[nx, ny] = rotulo
                                fila.append((nx, ny))
                
                rotulo += 1
    
    rotulos_unicos = numpy.unique(rotulado)
    mapeamento_colorido = mapeamento_cores(len(rotulos_unicos))
    
    rotulado_colorido = numpy.zeros((rotulado.shape[0], rotulado.shape[1], 3), dtype = numpy.uint8)
    
    for rotulo in rotulos_unicos:
        if rotulo == 0:
            continue
        rotulado_colorido[rotulado == rotulo] = mapeamento_colorido[rotulo]
    
    cv2.imwrite("watershed.png", rotulado_colorido)

if __name__ == "__main__":
    
# --------------- configuração do CLI ------------------------- #  
  
    parser = argparse.ArgumentParser(
        description = "Processamento de imagens utilizando diferentes transformadas."
    )
    parser.add_argument("-k", "--tamanho_kernel", type = int, default = 3,
        help = "Valor do kernel que será convertido para (w x w)."
    )
    parser.add_argument("-b", "--tamanho_banda", type = int, default = 3,
        help = "Largura da banda do filtro em pixels para os tipos de filtro passa-banda e rejeita-banda. Define a largura da faixa ativa ou inativa no filtro. Deve ser um valor inteiro."
    )
    parser.add_argument("-t", "--transformacao", type = str, default = "frequencia",
        help = "Transformacão que será aplicada na imagem"
    )
    parser.add_argument("-f", "--tipo_filtro", type = str, default = "passa baixa",
        help = "Opções de Filtragem: passa baixa, passa alta, passa banda e rejeita banda"
    )
    parser.add_argument("-r", "--raio", type = int, default = 3, 
        help = "Tamanho do filtro em pixels. Especifica o raio da região ativa no filtro passa-baixa, passa-alta, passa-banda ou rejeita-banda. Deve ser um valor inteiro."
    )
    parser.add_argument("-i", "--imagem", required = True,
        help = "Caminho da imagem de entrada."
    )
    parser.add_argument("-x", "--coordenada_x", type = int, default = 10,
        help = "Coordenada X da imagem"
    )
    parser.add_argument("-y", "--coordenada_y", type = int, default = 10,
        help = "Coordenada Y da imagem"
    )
    parser.add_argument("-l", "--limite", type = int, default = 127,
        help = "Limite do crescimento de região"
    )
    parser.add_argument("-v", "--limiar", type = int, default = 10,
        help = "Limiar do crescimento de região"
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
    elif args.transformacao == "regiao":
        crescimentoDeRegiao(imagem, args.coordenada_x, args.coordenada_y ,args.tamanho_kernel, args.limite)
    elif args.transformacao == "watershed":
        watershed(imagem, args.limiar, args.tamanho_kernel)
    else:
        print("Tipo de filtro inválido!")