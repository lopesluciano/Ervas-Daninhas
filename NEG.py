import cv2
import numpy as np

# Load the image
image = cv2.imread('ErvasDaninhas.png')
image = cv2.resize(image, (500, 400)) # Ajustando as Dimensoes

# Convert the image to float32 type
image_float = np.float32(image)

# Split the image into its channels
blue_channel, green_channel, red_channel = cv2.split(image_float)

# Normalize the channels
blue_channel_normalized = blue_channel / 255.0
green_channel_normalized = green_channel / 255.0
red_channel_normalized = red_channel / 255.0

# Calculate the excessive green (NEG) image
NEG = 2.8 * green_channel_normalized - red_channel_normalized - blue_channel_normalized

# Convert NEG pixel values to integer values by multiplying by 100
NEG_integer = (NEG * 100).astype(np.uint8)


# Display the normalized channels
cv2.imshow('Normalized Blue Channel', blue_channel_normalized)
cv2.imshow('Normalized Green Channel', green_channel_normalized)
cv2.imshow('Normalized Red Channel', red_channel_normalized)
cv2.imshow('NEG', NEG_integer)
cv2.waitKey(0)
cv2.destroyAllWindows()
