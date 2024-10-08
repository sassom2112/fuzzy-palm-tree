from PIL import Image

# Load the images
image_1_path = './2024-10-08 12_08_27-Single Neuron Classifier.ipynb - Colab.png'
image_2_path = './2024-10-08 12_11_20-Classifier_NLL_Loss.ipynb - Colab.png'
image_3_path = './2024-10-08 12_11_20-Classifier_NLL_Loss.ipynb - Colab.png'

images_1 = Image.open(image_1_path)
images_2 = Image.open(image_2_path)
images_3 = Image.open(image_3_path)

# Get the sizes of the images
size_1 = images_1.size  # (width, height)
size_2 = images_2.size
size_3 = images_3.size

# Calculate the smallest width and height among the images
smallest_width = min(size_1[0], size_2[0], size_3[0])
smallest_height = min(size_1[1], size_2[1], size_3[1])

# Set the target size to the smallest width and height
target_size = (smallest_width, smallest_height)

print(f"Target size: {target_size}")

# Resize the images if they are not already at target size
if images_1.size != target_size:
    images_1_resized = images_1.resize(target_size)
    images_1_resized.save(image_1_path)  # Overwrite the original image

if images_2.size != target_size:
    images_2_resized = images_2.resize(target_size)
    images_2_resized.save(image_2_path)  # Overwrite the original image

if images_3.size != target_size:
    images_3_resized = images_3.resize(target_size)
    images_3_resized.save(image_3_path)  # Overwrite the original image

# Check sizes after resizing
print("Resized image size 1: ", images_1_resized.size if images_1.size != target_size else images_1.size)
print("Resized image size 2: ", images_2_resized.size if images_2.size != target_size else images_2.size)
print("Resized image size 3: ", images_3_resized.size if images_3.size != target_size else images_3.size)
