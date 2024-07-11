import cv2
from ultralytics import YOLO

# Specify the model path (either "best.pt" or "last.pt")
model_path = '/home/luciano/Documents2/Ervas-Daninhas/weights/best.pt'  # or 'path/to/your/last.pt'

# Load your trained YOLOv8 model
model = YOLO(model_path)

# Load the image you want to analyze
image_path = '/home/luciano/Documents2/Ervas-Daninhas/Imagens/rgb_bonirob_2016-05-23-10-47-22_2_frame199.png'
image = cv2.imread(image_path)

# Perform the prediction
results = model.predict(image)

# Display the results
annotated_image = results[0].plot()

# Save or show the annotated image
output_path = 'path/to/save/annotated_image.jpg'
cv2.imwrite(output_path, annotated_image)
cv2.imshow('YOLOv8 Detection', annotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
