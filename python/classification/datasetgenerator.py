import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


PATCH_SIZE = (200, 200, 3)

IMAGE_DIR = "./garden"
TARGET_TEMPLATES_DIR = "./target_templates"


TARGET_EDGE_SIZE = 10


def main():
    generate_dataset()


def generate_dataset():
    images = read_all_images()

    plt.figure(1)
    while True:
        patch = get_random_image_patch(images)
        plt.imshow(patch)
        plt.pause(1)


def get_random_image_patch(images):
    image = random.choice(images)
    height, width, _ = image.shape

    ph, pw, _ = PATCH_SIZE

    y = random.randint(0, height - ph - 1)
    x = random.randint(0, width - pw - 1)

    patch = image[y:y + ph, x:x + pw, :]

    return np.flip(patch, axis=-1)


def get_targets():
    target_templates = []
    files = os.listdir(TARGET_TEMPLATES_DIR)
    for filename in files:
        path = os.path.join(TARGET_TEMPLATES_DIR, filename)
        target_templates.append(cv2.imread(path))

    h, w = target_templates[0].shape
    target_image = np.zeros((h + 2*TARGET_EDGE_SIZE, w + 2*TARGET_EDGE_SIZE, 3))

    num_squares_h = h // TARGET_EDGE_SIZE
    num_squares_w = w // TARGET_EDGE_SIZE

    for y in range(num_squares_h):
        for x in range(num_squares_w):
            if x > 1 or y > 1 or x < (num_squares_w - 1) or y < (num_squares_h - 1):
                continue
            # TODO create checker board edge


def read_all_images():
    images = []
    files = os.listdir(IMAGE_DIR)
    for filename in files:
        if filename.endswith(".bmp"):
            images.append(cv2.imread(os.path.join(IMAGE_DIR, filename)))
    return images


def create_checker_corner(image, y, x, h, w):
    image[y:y + h//2, x:x + w//2] = 1.
    image[y + h//2:y + h, x + w//2:x + w] = 1.


if __name__ == "__main__":
    main()
