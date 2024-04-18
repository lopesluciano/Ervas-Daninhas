import cv2
import numpy as np

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

# Bitwise-AND 
res = cv2.bitwise_and(frame, frame, mask=mask)

#Grayscale
gray_belt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

# Thresholding Simples
_, threshold = cv2.threshold(gray_belt, 50, 255, cv2.THRESH_BINARY)

# Metodo de Otsu
_, threshold2 = cv2.threshold(gray_belt, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Metodo de Otsu + Filtro 
blur = cv2.GaussianBlur(gray_belt, (5,5), 0)
_, threshold3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Detectar Contornos
contours, _ = cv2.findContours(threshold3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

for cnt in contours:
    (x, y, w, h) = cv2.boundingRect(cnt)

    # Calcula area
    area = cv2.contourArea(cnt)

    # Distinguir tamanho 
    if 5 < area < 1000:
        object_id = hash((x, y, w, h))  # Identificador do objeto na posicao
        if object_id not in tracked_objects:
            tracked_objects.add(object_id)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            total_counter += 1  # Soma o contador de objetos

    # Calcular o centroide
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                # Marcar o centroide
                cv2.circle(frame, (cX, cY), 2, (255, 0, 255), -1)

# Mostra o contador no quadro
#cv2.putText(frame, f"Total Count: {total_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


cv2.imshow("Frame", frame)
cv2.imshow("HSV", hsv_frame)
cv2.imshow("Verde Separado", res)
#cv2.imshow("Grayscale", gray_belt)
#cv2.imshow("Threshold", threshold)
#cv2.imshow("Otsu", threshold2)
#cv2.imshow("Otsu + Filtro Gaussiano", threshold3)

cv2.waitKey(0)
cv2.destroyAllWindows()