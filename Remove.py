import cv2
import numpy as np

# Carrega a imagem .png
frame = cv2.imread('Imagens/rgb_bonirob_2016-05-23-10-47-22_2_frame15.png')
frame = cv2.resize(frame, (500, 400))  # Ajustando as Dimensoes
    
# Converter quadro para HSV
hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Definir range de cor verde HSV
lower_green = np.array([25, 50, 2])
upper_green = np.array([85, 255, 255])

# Criar uma mascara para os pixels verdes
mask = cv2.inRange(hsv_frame, lower_green, upper_green)

# Aplicar a mascara na imagem original 
res = cv2.bitwise_and(frame, frame, mask=mask)

# Converter imagem resultante para escala cinza
gray_belt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# Metodo de Otsu para binarizacao
_, threshold2 = cv2.threshold(gray_belt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Detectar Contornos
contours, _ = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Criar uma mascara vazia para desenhar os contornos da cultura 
corn_mask = np.zeros_like(threshold2)

# Filtrar os contornos por area e desenhar os com maior area (caso base milho)
min_corn_area = 1500  # Valor de area minima (ajustavel)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_corn_area:
        cv2.drawContours(corn_mask, [contour], -1, (255), thickness=cv2.FILLED)

# Inverter a mascara da cultura para conseguir as ervas daninhas
weeds_mask = cv2.bitwise_not(corn_mask)

# Aplicar a mascara das ervas daninhas na imagem binarizada
weeds_only = cv2.bitwise_and(threshold2, threshold2, mask=weeds_mask)

# Remover pequenos objetos (Ruidos)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(weeds_only, connectivity=8)
min_size = 10  # Tamanho minimo para erva daninha
clean_weeds_only = np.zeros_like(weeds_only)

for i in range(1, num_labels):  # Inicia em 1
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        clean_weeds_only[labels == i] = 255

# Cria uma mascara separando as ervas daninhas na imagem original
weeds_original_mask = cv2.bitwise_and(frame, frame, mask=clean_weeds_only)

# Calcular os centroides das ervas daninhas
centroids_list = []
contours, _ = cv2.findContours(clean_weeds_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for contour in contours:
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        centroids_list.append((cX, cY))
        # Desenhar o centroide no quadro original
        cv2.circle(frame, (cX, cY), 3, (255, 0, 255), -1)

# Mostrar os resultados
cv2.imshow("Frame", frame)
cv2.imshow("Otsu", threshold2)
cv2.imshow('Weeds Only', clean_weeds_only)
cv2.imshow('Crop Only', corn_mask)
cv2.imshow('Weeds in Original Frame', weeds_original_mask)

print("Centroids of weeds:", centroids_list)

cv2.waitKey(0)
cv2.destroyAllWindows()
