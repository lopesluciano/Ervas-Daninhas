import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carrega a imagem .png
frame = cv2.imread('ErvasDaninhas.png')
frame = cv2.resize(frame, (500, 400)) # Ajustando as Dimensoes

# Inicializa contador de objetos
total_counter = 0
tracked_objects = set()

#Converter quadro para HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#Definir range de cor verde HSV
lower_green = np.array([25, 50, 50])
upper_green = np.array([85, 255, 255])

#Threshold HSV
mask = cv2.inRange(hsv_frame, lower_green, upper_green)

# Detectando Bordas utilizando o algoritmo de Canny
borda = cv2.Canny(mask,500,400)


# Dividindo em (H x W) regioes menores
height, width = mask.shape[:2]
region_height = height // 20
region_width = width // 20

# Inicializar Contadores
weed_regions = 0
total_region = 0

# limite de frequencia de borda
edge_threshold = 100

# Loop sobre cada regiao
for i in range(20):
    for j in range(20):
        # Coordenadas da regiao
        start_x = j * region_width
        end_x = (j + 1) * region_width
        start_y = i * region_height
        end_y = (i + 1) * region_height
        
        # Extract the region from the grayscale frame
        region = mask[start_y:end_y, start_x:end_x]
        
        # Apply Canny edge detection
        edges = cv2.Canny(region, 100, 200)
        
        # Count the number of edge pixels
        edge_count = np.count_nonzero(edges)
        
        # Determine if the region contains weeds based on edge frequency
        if edge_count > edge_threshold:  # Adjust threshold as needed
            # Draw a white square over the region
            frame = cv2.rectangle(frame, (start_x, start_y), (end_x, end_y), (255, 255, 255), -1)
            weed_regions += 1
        
        total_region += 1

# Display the result
cv2.imshow("Borda", borda)
cv2.imshow("edge", edges)
cv2.imshow("Result", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()