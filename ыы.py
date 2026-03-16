import os
from PIL import Image

input_root = r"PATH_TO_INPUT"
output_root = r"PATH_TO_OUTPUT"

positions = {
    "front": (0, 0),
    "side": (1, 0),
    "back": (0, 1),
    "selfie": (1, 1)
}

valid_ext = [".jpg", ".jpeg", ".png"]
output_size = 1024


def create_collage_from_folder(input_folder, output_path):
    cell = output_size // 2

    collage = Image.new("RGB", (output_size, output_size))
    black = Image.new("RGB", (cell, cell), (0, 0, 0))

    for x in range(2):
        for y in range(2):
            collage.paste(black, (x * cell, y * cell))

    for file in os.listdir(input_folder):
        name, ext = os.path.splitext(file)
        if ext.lower() not in valid_ext:
            continue

        name = name.lower()

        if name in positions:
            img_path = os.path.join(input_folder, file)
            img = Image.open(img_path).convert("RGB")
            img = img.resize((cell, cell))

            x, y = positions[name]
            collage.paste(img, (x * cell, y * cell))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    collage.save(output_path)


def process():
    for number_folder in os.listdir(input_root):
        number_path = os.path.join(input_root, number_folder)

        if not os.path.isdir(number_path):
            continue

        for sub in os.listdir(number_path):
            sub_path = os.path.join(number_path, sub)

            if not os.path.isdir(sub_path):
                continue

            output_sub_folder = os.path.join(output_root, number_folder, sub)
            output_file = os.path.join(output_sub_folder, "collage.jpg")

            create_collage_from_folder(sub_path, output_file)

            print(f"Сделано: {number_folder}/{sub}")


if __name__ == "__main__":
    process()
