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
lower_green = np.array([30, 30, 30])
upper_green = np.array([110, 255, 255])

#Threshold HSV
mask = cv2.inRange(hsv_frame, lower_green, upper_green)

# Bitwise-AND 
res = cv2.bitwise_and(frame, frame, mask=mask)

#Grayscale
gray_belt = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(gray_belt, 50, 255, cv2.THRESH_BINARY)

# Detectar Contornos
contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2:]

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

# Mostra o contador no quadro
#cv2.putText(frame, f"Total Count: {total_counter}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


cv2.imshow("Frame", frame)
cv2.imshow("HSV", hsv_frame)
cv2.imshow("Verde Separado", res)
cv2.imshow("Grayscale", gray_belt)
cv2.imshow("Threshold", threshold)

cv2.waitKey(0)
cv2.destroyAllWindows()
