import cv2
import numpy
import matplotlib.pyplot

import numpy as np
import matplotlib.pyplot as plt

def watershed_segmentation(image):
    # Conversão para escala de cinza
    gray = image.mean(axis=2)
    
    # Suavização da imagem
    blurred = np.sqrt(np.square(gray[:-2, :-2] - gray[2:, 2:]) + np.square(gray[:-2, 2:] - gray[2:, :-2]))
    
    # Binarização
    thresh = blurred > np.percentile(blurred, 95)
    
    # Transformada de distância
    def distance_transform(image):
        distance = np.zeros_like(image, dtype=float)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j]:
                    distance[i, j] = np.min(np.sqrt(np.square(i - np.arange(image.shape[0])) + np.square(j - np.arange(image.shape[1]))))
        return distance
    
    dist_transform = distance_transform(thresh)
    
    # Marcadores
    def label_image(image):
        labels = np.zeros_like(image, dtype=int)
        current_label = 1
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                if image[i, j]:
                    if labels[i, j] == 0:
                        stack = [(i, j)]
                        while stack:
                            x, y = stack.pop()
                            if 0 <= x < image.shape[0] and 0 <= y < image.shape[1] and image[x, y] and labels[x, y] == 0:
                                labels[x, y] = current_label
                                stack.extend([(x-1, y), (x+1, y), (x, y-1), (x, y+1)])
                        current_label += 1
        return labels
    
    markers = label_image(thresh)
    
    # Marcadores de fundo
    markers[0, :], markers[-1, :], markers[:, 0], markers[:, -1] = 0, 0, 0, 0
    
    # Atualização dos marcadores usando a transformada de distância
    for i in range(dist_transform.shape[0]):
        for j in range(dist_transform.shape[1]):
            if dist_transform[i, j] < np.percentile(dist_transform, 10):
                markers[i, j] = 0
    
    # Algoritmo de Watershed simplificado
    queue = np.zeros_like(markers, dtype=bool)
    queue[1:-1, 1:-1] = 1
    
    while np.any(queue):
        x, y = np.unravel_index(np.argmax(dist_transform * queue), queue.shape)
        queue[x, y] = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if markers[x + dx, y + dy] == 0 and dist_transform[x + dx, y + dy] < dist_transform[x, y]:
                    markers[x + dx, y + dy] = markers[x, y]
                    queue[x + dx, y + dy] = 1
    
    # Coloração das regiões segmentadas
    colors = np.random.randint(0, 255, size=(np.max(markers) + 1, 3))
    segmented_image = colors[markers]
    
    return segmented_image

# Exemplo de uso
image_path = 'path_to_image.jpg'
image = plt.imread(image_path)
segmented_image = watershed_segmentation(image)

# Exibição da imagem segmentada
plt.imshow(segmented_image)
plt.title('Segmented Image')
plt.axis('off')
plt.show()
