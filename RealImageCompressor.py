import os
import matplotlib.pyplot as plt
from PIL import Image
from keras.src.utils import load_img, img_to_array

# Set parameters
IMG_HEIGHT, IMG_WIDTH = 64, 64  # Target resolution (same as generator output)
dataset_path = './dataset/Data'  # Replace with your dataset path
category = 'Mild Dementia'  # Example category folder
num_images_to_display = 5  # Number of images to display

# Load and resize images
image_paths = os.listdir(os.path.join(dataset_path, category))[:num_images_to_display]
real_images = []

for image_name in image_paths:
    image_path = os.path.join(dataset_path, category, image_name)

    # Load and resize image
    image = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
    real_images.append(img_to_array(image))

# Display images
fig, axs = plt.subplots(1, num_images_to_display, figsize=(15, 5))
for i, img in enumerate(real_images):
    axs[i].imshow(img.squeeze(), cmap='gray')
    axs[i].axis('off')

plt.savefig('output_image.png')
