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
        filtro = numpy.zeros((linhas, colunas), numpy.uint8) - filtro
        
    elif tipo_filtro == 'rejeita-banda':
        filtro = numpy.ones((linhas, colunas), dtype=imagem.dtype)      
        filtro[centroLinha - tamanho : centroLinha + tamanho, centroColuna - tamanho - tamanho_banda : centroColuna + tamanho_banda] = 0

    imagem_filtrada_fft_shift = imagem_fft_shift * filtro
    
    imagem_filtrada_fft = numpy.fft.ifftshift(imagem_filtrada_fft_shift)
    imagem_filtrada = numpy.fft.ifft2(imagem_filtrada_fft)
    imagem_filtrada = numpy.abs(imagem_filtrada)
    
    imagem_filtrada = (255 * (imagem_filtrada - numpy.min(imagem_filtrada)) / (numpy.max(imagem_filtrada) - numpy.min(imagem_filtrada))).astype(numpy.uint8)    
    cv2.imwrite("filtragemFrequencia.png", imagem_filtrada)
    
# Questão 2

def erosao(imagem, kernel, foto):
    altura, largura = imagem.shape
    
    if isinstance(kernel, int): 
        kaltura = klargura = kernel
        kernel = numpy.ones((kaltura, klargura), dtype = int)
    else:
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
    if foto == 0:
        imagemErosada = saida.astype(numpy.uint8)
        cv2.imwrite("erosao.png", imagemErosada)
    return saida.astype(numpy.uint8)

def dilatacao(imagem, kernel, foto):
    altura, largura = imagem.shape
    
    if isinstance(kernel, int): 
        kaltura = klargura = kernel
        kernel = numpy.ones((kaltura, klargura), dtype = int)
    else:  
        kaltura, klargura = kernel.shape
        
    altura_pad = kaltura // 2
    largura_pad = klargura // 2
    
    saida = numpy.zeros((altura, largura), dtype = imagem.dtype)
    
    for i in range(altura_pad, altura - altura_pad):
        for j in range(largura_pad, largura - largura_pad):
            maximum = 0
            for m in range(kaltura):
                for n in range(klargura):
                    if kernel[m, n] == 1:
                        maximum = max(maximum, imagem[i - altura_pad + m, j - largura_pad + n])
            saida[i, j] = maximum
    if foto == 0:
        imagemDilatada = saida.astype(numpy.uint8)
        cv2.imwrite("dilatada.png", imagemDilatada)
    return saida.astype(numpy.uint8)

# Questão 3

def limiarizacaoMediaMovel(imagem, tamanho_janela):

    altura, largura = imagem.shape
    saida = numpy.zeros((altura, largura), dtype = imagem.dtype)
    
    for i in range(altura):
        for j in range(largura):
            inicioAltura = max(0, i - tamanho_janela // 2)
            fimAltura = min(altura, i + tamanho_janela // 2)
            inicioLargura = max(0, j - tamanho_janela // 2)
            fimLargura = min(largura, j + tamanho_janela // 2)
            
            media = numpy.mean(imagem[inicioAltura : fimAltura, inicioLargura : fimLargura])
            
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
    rotulada = numpy.zeros_like(imagem, dtype = numpy.int32)
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
    return numpy.random.randint(0, 256, size=(num_rotulos, 3), dtype = numpy.uint8)

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

def aplicar_abertura_manual(imagem, kernel):
    imagem_erodida = erosao(imagem, kernel, 1)
    imagem_abertura = dilatacao(imagem_erodida, kernel, 1)
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
    
# Questão 6
def desfoqueGaussiano(imagem, tamanho_kernel):
    kernel = numpy.outer(
        numpy.exp(-numpy.linspace(-1, 1, tamanho_kernel) ** 2),
        numpy.exp(-numpy.linspace(-1, 1, tamanho_kernel) ** 2)
    )
    kernel /= kernel.sum()

    suavizada = cv2.filter2D(imagem, -1, kernel)
    return suavizada.astype(numpy.uint8)

def gradiente(imagem):
    kernel_x = numpy.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    kernel_y = numpy.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    gradiente_x = cv2.filter2D(imagem, -1, kernel_x)
    gradiente_y = cv2.filter2D(imagem, -1, kernel_y)

    magnitude_gradiente = numpy.sqrt(gradiente_x ** 2 + gradiente_y ** 2)
    angulo_gradiente = numpy.arctan2(gradiente_y, gradiente_x)

    return magnitude_gradiente, angulo_gradiente

def supressao(magnitude_gradiente, angulo_gradiente):
    
    suprimida = numpy.zeros_like(magnitude_gradiente)
    direcoes = numpy.rad2deg(angulo_gradiente) % 180

    for i in range(1, magnitude_gradiente.shape[0] - 1):
        for j in range(1, magnitude_gradiente.shape[1] - 1):
            direcao = direcoes[i, j]
            if direcao == 0:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j + 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i, j - 1])
            elif direcao == 45:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j + 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j - 1])
            elif direcao == 90:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j])
            elif direcao == 135:
                suprimida[i, j] = (magnitude_gradiente[i, j] >= magnitude_gradiente[i - 1, j - 1]) and \
                                  (magnitude_gradiente[i, j] >= magnitude_gradiente[i + 1, j + 1])

    return suprimida * magnitude_gradiente

def limiarOtsu(imagem):
    hist, _ = numpy.histogram(imagem.flatten(), bins=256, range=[0,256])

    probabilidade = hist / float(numpy.sum(hist))

    melhor_limiar = 0
    melhor_variancia_intra = 0

    for t in range(1, 256):
        w0 = numpy.sum(probabilidade[:t])
        w1 = numpy.sum(probabilidade[t:])

        u0 = numpy.sum(numpy.arange(t) * probabilidade[:t]) / w0 if w0 > 0 else 0
        u1 = numpy.sum(numpy.arange(t, 256) * probabilidade[t:]) / w1 if w1 > 0 else 0

        variancia_intra = w0 * w1 * ((u0 - u1) ** 2)

        if variancia_intra > melhor_variancia_intra:
            melhor_variancia_intra = variancia_intra
            melhor_limiar = t

    limiarizado = numpy.zeros_like(imagem)
    limiarizado[imagem >= melhor_limiar] = 255

    return limiarizado


def canny(imagem):
    suavizada = desfoqueGaussiano(imagem, tamanho_kernel=5)
    gradientes, angulos = gradiente(suavizada)
    suprimida = supressao(gradientes, angulos)
    bordas = limiarOtsu(suprimida)
    
    return bordas


def houghLinhas(imagem, limiar):
    bordas = canny(imagem)
    
    altura, largura = bordas.shape
    diagonal = numpy.ceil(numpy.sqrt(altura ** 2 + largura ** 2))
    thetas = numpy.deg2rad(numpy.arange(-90, 90))
    cos_t = numpy.cos(thetas)
    sin_t = numpy.sin(thetas)
    num_thetas = len(thetas)
    
    acumulador = numpy.zeros((2 * int(diagonal), num_thetas), dtype=numpy.uint64)
    y_idxs, x_idxs = numpy.nonzero(bordas)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for t_idx in range(num_thetas):
            rho = int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + int(diagonal)  # Converter rho para inteiro

            acumulador[rho, t_idx] += 1

    rhos, thetas = numpy.where(acumulador >= limiar)
    result_imagem = imagem.copy()
    
    for rho, theta in zip(rhos, thetas):
        a = numpy.cos(thetas[theta])
        b = numpy.sin(thetas[theta])
        x0 = a * (rho - diagonal)
        y0 = b * (rho - diagonal)
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(result_imagem, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Linhas em azul

    cv2.imwrite("linha.png", result_imagem)

def houghCirculos(imagem, raio, limiar):
    bordas = canny(imagem)
    
    altura, largura = bordas.shape
    acumulador = numpy.zeros((altura, largura, raio), dtype=numpy.uint64)
    
    thetas = numpy.linspace(0, 2*numpy.pi, 360)
    cos_t = numpy.cos(thetas)
    sin_t = numpy.sin(thetas)
    
    y_idxs, x_idxs = numpy.nonzero(bordas)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]

        for theta_idx in range(len(thetas)):
            for r in range(raio):
                a = int(x - r * cos_t[theta_idx])
                b = int(y - r * sin_t[theta_idx])

                if 0 <= a < largura and 0 <= b < altura:
                    acumulador[b, a, r] += 1
    
    y_pico, x_pico, r_pico = numpy.where(acumulador >= limiar)
    result_imagem = imagem.copy()
    
    for i in range(len(x_pico)):
        cv2.circle(result_imagem, (x_pico[i], y_pico[i]), raio, (0, 0, 255), 2)

    cv2.imwrite("circulo.png", result_imagem)

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
        erosao(imagem, args.tamanho_kernel, 0)
    elif args.transformacao == "dilatacao":
        dilatacao(imagem, args.tamanho_kernel, 0)
    elif args.transformacao == "movel":
        limiarizacaoMediaMovel(imagem, args.tamanho_kernel)
    elif args.transformacao == "regiao":
        crescimentoDeRegiao(imagem, args.coordenada_x, args.coordenada_y ,args.tamanho_kernel, args.limite)
    elif args.transformacao == "watershed":
        watershed(imagem, args.limiar, args.tamanho_kernel)
    elif args.transformacao == "hough":
        houghLinhas(imagem, args.limiar)
        houghCirculos(imagem, args.raio, args.limiar)
    else:
        print("Tipo de filtro inválido!")