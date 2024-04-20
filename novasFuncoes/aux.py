import cv2
import numpy as np
from matplotlib import pyplot as plt

def gerar_imagem_aleatoria(formato, altura, largura):
    if formato == "RGB":
        imagem = np.random.randint(0, 256, size=(altura, largura, 3), dtype=np.uint8)
    elif formato == "RGBA":
        imagem = np.random.randint(0, 256, size=(altura, largura, 4), dtype=np.uint8)
    elif formato == "Tons de Cinza":
        imagem = np.random.randint(0, 256, size=(altura, largura), dtype=np.uint8)
    elif formato == "Binaria":
        imagem = np.random.choice([0, 255], size=(altura, largura))
    else:
        raise ValueError("Formato de imagem inválido.")
    return imagem

def converter_rgb_para_cmyk(imagem_rgb):
    r = imagem_rgb[:, :, 0] / 255.0
    g = imagem_rgb[:, :, 1] / 255.0
    b = imagem_rgb[:, :, 2] / 255.0

    k = np.minimum(np.minimum(1 - r, 1 - g), 1 - b)
    c = (1 - r - k) / (1 - k)
    m = (1 - g - k) / (1 - k)
    y = (1 - b - k) / (1 - k)

    imagem_cmyk = np.zeros((imagem_rgb.shape[0], imagem_rgb.shape[1], 4), dtype=np.uint8)
    imagem_cmyk[:, :, 0] = c * 255  # Cyan
    imagem_cmyk[:, :, 1] = m * 255  # Magenta
    imagem_cmyk[:, :, 2] = y * 255  # Yellow
    imagem_cmyk[:, :, 3] = k * 255  # Black

    return imagem_cmyk

# Definir altura e largura
altura = 20
largura = 20

# Gerar imagens aleatórias
imagem_rgb = gerar_imagem_aleatoria("RGB", altura, largura)
imagem_rgba = gerar_imagem_aleatoria("RGBA", altura, largura)
imagem_tons_de_cinza = gerar_imagem_aleatoria("Tons de Cinza", altura, largura)
imagem_binaria = gerar_imagem_aleatoria("Binaria", altura, largura)

# Converter RGB para CMYK
imagem_cmyk = converter_rgb_para_cmyk(imagem_rgb)

# Exibir imagens
plt.subplot(231)
plt.imshow(imagem_rgb)
plt.title("RGB")

plt.subplot(232)
plt.imshow(imagem_rgba)
plt.title("RGBA")

plt.subplot(233)
plt.imshow(imagem_tons_de_cinza, cmap="gray")
plt.title("Tons de Cinza")

plt.subplot(234)
plt.imshow(imagem_binaria)
plt.title("Binaria")

plt.subplot(235)
plt.imshow(imagem_cmyk[:, :, :3])
plt.title("CMYK")

plt.show()

cv2.imwrite("imagem_rgb.png", imagem_rgb)
cv2.imwrite("imagem_rgba.png", imagem_rgba)
cv2.imwrite("imagem_tons_de_cinza.png", imagem_tons_de_cinza)
cv2.imwrite("imagem_binaria.png", imagem_binaria)
cv2.imwrite("imagem_cmyk.png", imagem_cmyk[:, :, :3])

print("Imagens salvas com sucesso!")
