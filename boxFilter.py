import numpy
import cv2
import sys
import getopt

def lerImagem(arquivo):
    
    # Ler um arquivo de imagem, eliminar erros se não conseguirmos encontrar o arquivo
    # :param arquivo: O nome do arquivo da imagem.
    # :return: O objeto img em forma de matriz.
    
    imgem = cv2.imread(arquivo)
    if imgem is None:
        print(f'Imagem Invalida: {arquivo}')
        return None
    else:
        print(f'Imagem Valida')
        return imgem

def integralDaImagem(imagem):
    
    # Retorna a tabela de imagem integral/área somada.
    # :param imagem:
    # :return:
    
    altura = imagem.shape[0]
    largura = imagem.shape[1]
    # int_imagem = numpy.zeros((altura, largura), numpy.uint64)
    int_imagem = numpy.zeros((altura, largura, 3), dtype = numpy.uint64)
    for y in range(altura):
        for x in range(largura):
            if (y - 1 < 0): 
                acima = 0
            else: 
                acima = int_imagem[y - 1, x]
                # acima = int_imagem.item((y - 1, x))
                
            if (x - 1 < 0):
                esquerda = 0
            else:
                # esquerda = int_imagem.item((y, x - 1))
                esquerda = int_imagem[y, x - 1]
            
            if (x - 1 < 0 or y - 1 < 0):
                diagonal = 0
            else:
                diagonal = int_imagem[y - 1, x - 1]
                #diagonal = int_imagem.item((y - 1, x - 1))
                
            #resultado = imagem.item((y, x)) + int(acima) + int(esquerda) - int(diagonal)
            #resultado = int(imagem[y, x]) + int(acima) + int(esquerda) - int(diagonal)
            resultado = imagem[y, x] + acima + esquerda - diagonal

            int_imagem.itemset((y, x), resultado)
            #int_imagem[(y, x), resultado]
            
    return int_imagem

def ajustarBordas(altura, largura, ponto):
    
    # Função para tratar dos casos extremos quando os limites da caixa estiverem fora do 
    # intervalo da imagem com base no pixel atual.    
    # :param altura: Altura da imagem.
    # :param largura: Largura da imagem.
    # :param ponto: O ponto atual.
    # :return:
    
    novoPonto = [ponto[0], ponto[1]]
    if ponto[0] >= altura:
        novoPonto[0] = altura - 1

    if ponto[1] >= largura:
        novoPonto[1] = largura - 1
    return tuple(novoPonto)

def encontrarArea(imagem, cantoSuperiorEsq, cantoSuperiorDir, cantoInferiorEsq, cantoInferiorDir):
    
    altura = imagem.shape[0]
    largura = imagem.shape[1]
    
    cantoSuperiorEsq = ajustarBordas(altura, largura, cantoSuperiorEsq)
    cantoSuperiorDir = ajustarBordas(altura, largura, cantoSuperiorDir)
    cantoInferiorEsq = ajustarBordas(altura, largura, cantoInferiorEsq)
    cantoInferiorDir = ajustarBordas(altura, largura, cantoInferiorDir)

    if (cantoSuperiorEsq[0] < 0 or cantoSuperiorEsq[0] >= altura) or (cantoSuperiorEsq[1] < 0 or cantoSuperiorEsq[1] >= largura):
        cantoSuperiorEsq = 0
    else: 
        cantoSuperiorEsq = imagem.item(cantoSuperiorEsq[0], cantoSuperiorEsq[1])
    
    if (cantoSuperiorDir[0] < 0 or cantoSuperiorDir[0] >= altura) or (cantoSuperiorDir[1] < 0 or cantoSuperiorDir[1] >= largura):
        cantoSuperiorDir = 0 
    else: 
        cantoSuperiorDir = imagem.item(cantoSuperiorDir[0], cantoSuperiorDir[1])
        
    if (cantoInferiorEsq[0] < 0 or cantoInferiorEsq[0] >= altura) or (cantoInferiorEsq[1] < 0 or cantoInferiorEsq[1] >= largura):
        cantoInferiorEsq = 0
    else: 
        cantoInferiorEsq = imagem.item(cantoInferiorEsq[0], cantoInferiorEsq[1])
        
    if (cantoInferiorDir[0] < 0 or cantoInferiorDir[0] >= altura) or (cantoInferiorDir[1] < 0 or cantoInferiorDir[1] >= largura):
        cantoInferiorDir = 0
    else: 
        cantoInferiorDir = imagem.item(cantoInferiorDir[0], cantoInferiorDir[1])

    return cantoSuperiorEsq + cantoInferiorDir - cantoSuperiorDir - cantoInferiorEsq

def filtroDaMedia(imagem, filtro):
    
    # Imprime a imagem original, encontra a imagem integral e depois gera a imagem final   
    # :param imagem: Uma imagem em forma de matriz.
    # :param filtro: O tamanho do filtro da matriz.
    # :return: Uma imagem escrita como imagem.png
    
    print("Imagem Original")
    print(imagem)
    
    altura = imagem.shape[0]
    largura = imagem.shape[1]
    imagemIntegral = integralDaImagem(imagem)
    imagemFinal = numpy.ones((altura, largura), numpy.uint64)
    
    print("Imagem Integral")
    print(imagemIntegral)
    cv2.imwrite("imagemIntegral.png", imagemIntegral)
    
    media = filtro/2
    for y in range(altura):
        for x in range(largura):
            imagemFinal.itemset(
                (y, x), 
                encontrarArea(
                    imagemIntegral, 
                    (y - media - 1, x - media - 1),
                    (y - media - 1, x + media),
                    (y + media, x - media - 1),
                    (y + media, x + media)
                ) / (filtro ** 2)
            )
    print("Imagem Final")
    print(imagemFinal)

    cv2.imwrite("imagemFinal.png", imagemFinal)

def main():
    
    # Ler a imagem e lida com a análise de argumentos
    # :return: None

    args, img_name = getopt.getopt(sys.argv[1:], '', ['filter_size='])
    args = dict(args)
    filter_size = args.get('--filter_size')

    print(f"Imagem: {img_name[0]}")
    print(f"Filtro: {filter_size}")

    imagem = lerImagem(img_name[0])
    if imagem is not None:
        print (f"Shape: {imagem.shape}")
        print (f"Size: {imagem.size}")
        print (f"Type: {imagem.dtype}")
        filtroDaMedia(imagem, int(filter_size))


if __name__ == "__main__":
    main()