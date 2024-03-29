# VC-IFCE

## Processamento de Imagens com Filtros

### Este é um script Python que permite o processamento de imagens utilizando diferentes filtros. O script utiliza a biblioteca OpenCV (cv2) e numpy para manipular e processar as imagens.

#### Requisitos

* Python 3.x
* OpenCV (cv2)
* Numpy

#### Como usar o script

O script aceita os seguintes parâmetros através da linha de comando:

* -w, --tamanho_janela: Define o tamanho da janela para os filtros. Deve ser um número inteiro positivo. O valor padrão é 3.

* -m, --mascara: Caminho para um arquivo .txt contendo a máscara utilizada no filtro Laplaciano. Este parâmetro é obrigatório apenas para o filtro Laplaciano.

* -f, --tipo_filtro: Define o tipo de filtro a ser aplicado. As opções disponíveis são:
- media: Filtro da média
- mediana: Filtro da mediana
- gaussiano: Filtro Gaussiano
- laplaciano: Filtro Laplaciano
- prewitt: Filtro de Prewitt
- sobel: Filtro de Sobel

* -s, --sigma: Valor do sigma utilizado exclusivamente no filtro gaussiano. Deve ser um número float. O valor padrão é 1.0.
* -i, --imagem: Caminho da imagem de entrada que será processada.
#### Exemplos de Uso

##### Filtro da Média
* python3 filtros.py -f "media" -w 6 -i caminho_da_imagem.png
##### Filtro da Mediana
* python3 filtros.py -f "mediana" -w 5 -i caminho_da_imagem.png
##### Filtro Gaussiano
* python filtros.py -f "gaussiano" -w 4 -s 1.756 -i caminho_da_imagem.png
##### Filtro Laplaciano
* Para utilizar o filtro Laplaciano, é necessário fornecer um arquivo .txt contendo a máscara. Por exemplo, uma máscara válida é:

0 -1 0
-1 4 -1
0 -1 0

A máscara deve ser quadrada

Salve esta máscara em um arquivo mascara.txt e execute:
* python filtros.py -f "laplaciano" -w 4 -m "mascara.txt" -i caminho_da_imagem.png

##### Filtro de Prewitt
* python filtros.py -f "prewitt" -i caminho_da_imagem.png
##### Filtro de Sobel
* python nome_do_script.py -f sobel -i caminho_da_imagem.png

#### Saída

O script salvará a imagem processada no diretório atual com o nome baseado no tipo de filtro utilizado. Por exemplo, se você usar o filtro da média, a imagem resultante será media.png.

#### Exemplo Completo

Para aplicar o filtro Gaussiano com uma janela de tamanho 5 e sigma 2.0 em uma imagem chamada imagem.jpg, você usaria o seguinte comando:

python filtros.py -f gaussiano -w 5 -s 2.0 -i imagem.jpg
Após executar este comando, a imagem processada será salva como gaussiana.png no diretório atual.

