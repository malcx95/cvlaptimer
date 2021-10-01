import numpy as np
import cv2
import os
import random
import matplotlib.pyplot as plt


PATCH_SIZE = (200, 200, 3)

IMAGE_DIR = "./garden"
TARGET_TEMPLATES_DIR = "./target_templates"


TARGET_EDGE_SIZE = 20


def main():
    #_plot_targets()
    #generate_dataset()

    targets = get_targets()
    _random_perspective_distortion(targets['A'])


def _plot_targets():
    plt.figure(1)
    targets = get_targets()
    for i, (label, target) in enumerate(targets.items()):
        plt.subplot(2, 3, i + 1)
        plt.imshow(target)
        plt.title(label)

    plt.show()


def generate_dataset():
    images = read_all_images()
    targets = get_targets()

    plt.figure(1)
    while True:
        patch = generate_random_ground_truth_image(targets, images)
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


def generate_random_ground_truth_image(targets, background_images):
    pass


def randomly_distort_target_image(target_image):
    pass


def _random_perspective_distortion(target_image):
    plt.figure(1)
    #plt.subplot(4, 4, 1)

    while True:
        h, w, _ = target_image.shape

        g = float(random.randint(0, 1000) - 500) / 5e5
        h = float(random.randint(0, 1000)- 500) / 5e5


        translation1 = np.array([
            [1., 0., -h/2],
            [0., 1., -w/2],
            [0., 0., 1.],
        ])

        matrix = np.array([
            [1., 0., 0.],
            [0., 1., 0.],
            [h, g, 1.],
        ])

        translation2 = np.array([
            [1., 0., h/2],
            [0., 1., w/2],
            [0., 0., 1.],
        ])

        total_transformation = np.matmul(np.matmul(translation1, matrix), translation2)
        total_transformation = total_transformation / total_transformation[2, 2]
        #warped1 = cv2.warpPerspective(target_image, matrix, target_image.shape[:2])

        #plt.subplot(1, 2, 1)
        #plt.imshow(warped1)

        warped2 = cv2.warpPerspective(target_image, total_transformation, target_image.shape[:2])
        #plt.subplot(1, 2, 2)
        plt.imshow(warped2)
        plt.title(f"h={h:.5f}\ng={g:.5f}")

        plt.pause(1)
        plt.clf()


def get_targets():
    target_templates = {}
    files = os.listdir(TARGET_TEMPLATES_DIR)
    for filename in files:
        path = os.path.join(TARGET_TEMPLATES_DIR, filename)
        target_templates[filename[0].upper()] = cv2.imread(path)

    h, w, _ = list(target_templates.values())[0].shape
    target_image = np.zeros((h + 2*TARGET_EDGE_SIZE, w + 2*TARGET_EDGE_SIZE, 3), dtype=np.float32)

    th, tw, _ = target_image.shape

    num_squares_h = th // TARGET_EDGE_SIZE
    num_squares_w = tw // TARGET_EDGE_SIZE

    for y in range(num_squares_h):
        for x in range(num_squares_w):
            if x in (0, num_squares_w - 1) or y in (0, num_squares_h - 1):
                # this will create alternating squares
                color = float((y % 2) ^ (x % 2))
                target_image[y*TARGET_EDGE_SIZE:(y + 1)*TARGET_EDGE_SIZE,
                             x*TARGET_EDGE_SIZE:(x + 1)*TARGET_EDGE_SIZE, :] = color

    target_images = {}
    for label, template in target_templates.items():
        im = np.copy(target_image)
        im[TARGET_EDGE_SIZE:h + TARGET_EDGE_SIZE,
           TARGET_EDGE_SIZE:w + TARGET_EDGE_SIZE] = template.astype(np.float32) / 255.
        target_images[label] = im

    return target_images


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
    try:
        main()
    except KeyboardInterrupt:
        pass
