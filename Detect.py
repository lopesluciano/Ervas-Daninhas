import cv2
import numpy as np

# Carrega a imagem .png
frame = cv2.imread('Imagens/Fig2UofAntioquiaArticle.png')

    
frame = cv2.resize(frame, (500, 400))  # Ajustando as Dimensoes


## Separacao do verde da imagem ##  

# Converter a imagem de BGR para HSV
hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Definindo uma região de interesse (ROI) onde se sabe que é verde
roi = hsv_image  # Ajuste os valores para a sua imagem

# Calcular média e desvio padrão
mean = cv2.mean(roi)  # mean retorna um tuple com 4 elementos (média e 0.0 para a quarta entrada)
mean = np.array(mean[:3])  # Pega apenas os três primeiros valores (H, S, V)
stddev = cv2.meanStdDev(roi)[1].flatten()  # Retorna um array 2D, então usamos flatten() para torná-lo 1D

# Definindo os limites com base na média e no desvio padrão
lower_green = np.maximum(0, mean - stddev)
upper_green = np.minimum([180, 255, 255], mean + stddev)  # 180 é o valor máximo para Hue

# Criar uma máscara para a cor verde
mask = cv2.inRange(hsv_image, lower_green.astype(int), upper_green.astype(int))

# Aplicar a mascara na imagem original 
res = cv2.bitwise_and(frame, frame, mask=mask)
cv2.imshow("res", res)

# Converter imagem resultante para escala cinza
gray_belt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray_belt", gray_belt)



## Filtro Medio ##
#plant_image_filtered = cv2.medianBlur(gray_belt, 3)
#cv2.imshow("plant_plant_image_filtered", plant_image_filtered)


## Segementacao da Imagem ##


# Metodo de Otsu para binarizacao
_, threshold2 = cv2.threshold(gray_belt, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu", threshold2)


## Classificacao por area ##

# Detectar Contornos
contours, _ = cv2.findContours(threshold2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Criar uma mascara vazia para desenhar os contornos da cultura 
crop_mask = np.zeros_like(threshold2)
cv2.imshow('Crop Only', crop_mask)

# Filtrar os contornos por area e desenhar os com maior area (caso base milho)
min_crop_area = 1500  # Valor de area minima (ajustavel)
for contour in contours:
    area = cv2.contourArea(contour)
    if area > min_crop_area:
        cv2.drawContours(crop_mask, [contour], -1, (255), thickness=cv2.FILLED)

# Inverter a mascara da cultura para conseguir as ervas daninhas
weeds_mask = cv2.bitwise_not(crop_mask)

# Aplicar a mascara das ervas daninhas na imagem binarizada
weeds_only = cv2.bitwise_and(threshold2, threshold2, mask=weeds_mask)

# Remover pequenos objetos (Ruidos)
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(weeds_only, connectivity=8)
min_size = 10  # Tamanho minimo para erva daninha
clean_weeds_only = np.zeros_like(weeds_only)
cv2.imshow('Weeds Only', clean_weeds_only)

for i in range(1, num_labels):  # Inicia em 1
    if stats[i, cv2.CC_STAT_AREA] >= min_size:
        clean_weeds_only[labels == i] = 255

# Cria uma mascara separando as ervas daninhas na imagem original
weeds_original_mask = cv2.bitwise_and(frame, frame, mask=clean_weeds_only)
cv2.imshow('Weeds in Original Frame', weeds_original_mask)

# Cria uma mascara separando a cultura na imagem original
crop_original_mask = cv2.bitwise_and(frame, frame, mask=crop_mask)
cv2.imshow('Crops in Original Frame', crop_original_mask)

## Calculo dos centroides ##

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

print("Centroids of weeds:", centroids_list)

cv2.waitKey(0)
cv2.destroyAllWindows()

