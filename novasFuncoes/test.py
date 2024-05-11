import numpy as np
import cv2
from matplotlib import pyplot as plt

def processarImagem(imagem, metodo, filtro, kernel, forma, iteracoes):
    
    # Carrega a imagem em tons de cinza
    img = cv2.imread(imagem, 0)

    # Filtragem espacial
    if metodo == 'spatial':
        if filtro == 'gaussian':
            filtered_img = cv2.GaussianBlur(img, (kernel, kernel), 0)
        elif filtro == 'median':
            filtered_img = cv2.medianBlur(img, kernel)
        elif filtro == 'laplacian':
            filtered_img = cv2.Laplacian(img, cv2.CV_64F, ksize = kernel)
        else:
            raise ValueError("Invalid filter type for spatial filtering.")
    # Filtragem na frequência
    elif metodo == 'frequency':
        dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

        # Criação de um filtro Gaussiano no domínio da frequência
        rows, cols = img.shape
        crow, ccol = rows / 2, cols / 2
        mask = np.zeros((rows, cols, 2), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        filtered_img = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    else:
        raise ValueError("Invalid metodo selected.")

    # Erosão e Dilatação
    if forma == 'rectangle':
        kernel = np.ones((kernel, kernel), np.uint8)
    elif forma == 'ellipse':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel, kernel))
    else:
        raise ValueError("Invalid kernel shape.")

    eroded_img = cv2.erode(filtered_img, kernel, iterations = iteracoes)
    dilated_img = cv2.dilate(filtered_img, kernel, iterations = iteracoes)

    # Plotar as imagens
    plt.subplot(221), plt.imshow(img, cmap='gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(222), plt.imshow(filtered_img, cmap='gray')
    plt.title('Filtered'), plt.xticks([]), plt.yticks([])
    plt.subplot(223), plt.imshow(eroded_img, cmap='gray')
    plt.title('Eroded'), plt.xticks([]), plt.yticks([])
    plt.subplot(224), plt.imshow(dilated_img, cmap='gray')
    plt.title('Dilated'), plt.xticks([]), plt.yticks([])
    plt.show()

# Exemplo de uso:
processarImagem('trabalho.png', metodo='spatial', filtro='gaussian', kernel=5, forma='rectangle', iteracoes=1)


def apply_frequency_filter(image_path, filter_type, filter_params, save_path):
    """
    Aplica um filtro no domínio da frequência em uma imagem e salva a imagem filtrada.

    Parâmetros:
    - image_path (str): Caminho da imagem a ser processada.
    - filter_type (str): Tipo de filtro a ser aplicado ('passa_alta', 'passa_baixa', 'passa_banda', 'rejeita_banda').
    - filter_params (dict): Parâmetros do filtro (dependendo do tipo de filtro).
    - save_path (str): Caminho onde a imagem filtrada será salva.
    """

    # Carrega a imagem em tons de cinza
    img = cv2.imread(image_path, 0)

    # Calcula a DFT da imagem
    dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = img.shape
    crow, ccol = rows // 2, cols // 2

    # Cria a máscara do filtro
    mask = np.zeros((rows, cols, 2), np.uint8)

    if filter_type == 'passa_alta':
        mask[crow - filter_params['radius']:crow + filter_params['radius'], 
             ccol - filter_params['radius']:ccol + filter_params['radius']] = 1

    elif filter_type == 'passa_baixa':
        mask[crow - filter_params['radius']:crow + filter_params['radius'], 
             ccol - filter_params['radius']:ccol + filter_params['radius']] = 1
        mask = 1 - mask

    elif filter_type == 'passa_banda':
        mask[crow - filter_params['inner_radius']:crow + filter_params['inner_radius'], 
             ccol - filter_params['inner_radius']:ccol + filter_params['inner_radius']] = 1
        mask[crow - filter_params['outer_radius']:crow + filter_params['outer_radius'], 
             ccol - filter_params['outer_radius']:ccol + filter_params['outer_radius']] = 0

    elif filter_type == 'rejeita_banda':
        mask[crow - filter_params['inner_radius']:crow + filter_params['inner_radius'], 
             ccol - filter_params['inner_radius']:ccol + filter_params['inner_radius']] = 0
        mask[crow - filter_params['outer_radius']:crow + filter_params['outer_radius'], 
             ccol - filter_params['outer_radius']:ccol + filter_params['outer_radius']] = 1

    # Aplica o filtro
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_filtered = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

    # Salva a imagem filtrada
    cv2.imwrite(save_path, img_filtered)

# Exemplo de uso:
filter_params = {
    'radius': 30,
    'inner_radius': 20,
    'outer_radius': 40
}
apply_frequency_filter('trabalho.png', 'passa_alta', filter_params, 'filtered_image.jpg')
