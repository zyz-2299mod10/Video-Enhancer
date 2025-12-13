from PIL import Image
import os

image_folder = './concate_img'
os.makedirs(image_folder, exist_ok=True)

image_files = sorted([
    f for f in os.listdir(image_folder)
    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))
])

images = []
max_size = 0

for f in image_files:
    img = Image.open(os.path.join(image_folder, f))
    w, h = img.size
    max_size = max(max_size, w, h)
    images.append(img)

images = [img.resize((max_size, max_size)) for img in images]

images_per_row = 2
num_images = len(images)
num_rows = (num_images + images_per_row - 1) // images_per_row

total_width = max_size * images_per_row
total_height = max_size * num_rows

new_image = Image.new('RGB', (total_width, total_height), color=(0, 0, 0))

for idx, img in enumerate(images):
    row = idx // images_per_row
    col = idx % images_per_row

    x_offset = col * max_size
    y_offset = row * max_size
    new_image.paste(img, (x_offset, y_offset))

new_image.save('./concate_img/concatenated_result.png')
new_image.show()
