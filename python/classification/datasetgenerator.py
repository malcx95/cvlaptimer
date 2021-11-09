import numpy as np
import scipy.signal as signal
import cv2
import os
import random
import matplotlib.pyplot as plt


PATCH_SIZE = (200, 200, 3)

IMAGE_DIR = "./garden"
TARGET_TEMPLATES_DIR = "./target_templates"


TARGET_EDGE_SIZE = 20

MAX_ROTATION_ANGLE_DEG = 25.


def main():
    targets = get_targets()
    _plot_random_targets(targets)


def _plot_random_targets(targets):
    plt.figure(1)

    while True:
        warped = _random_distortion(targets['A'])
        blurred, kernel = _add_random_blur(warped)
        plt.imshow(blurred)
        plt.pause(2)
        plt.clf()


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
        plt.pause(2)


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


def _add_random_blur(target_image):
    kernel_size = 10
    blur_amount, = np.abs(np.random.normal(0.0001, 1., 1))
    kernel = _get_gaussian_kernel(kernel_size, blur_amount)
    motion_blur_amount, = np.random.uniform(0.5, 1., 1)
    motion_blur_angle, = np.random.uniform(0., np.pi, 1)
    translation = np.array([
        [1., 0., kernel_size/2],
        [0., 1., kernel_size/2],
        [0., 0., 1.],
    ])
    inverse_translation = np.array([
        [1., 0., -kernel_size/2],
        [0., 1., -kernel_size/2],
        [0., 0.,  1.],
    ])
    scale_matrix = np.array([
        [1., 0., 0.],
        [0., motion_blur_amount, 0.],
        [0., 0., 1.],
    ])
    rotation_matrix = np.array([
        [np.cos(motion_blur_angle), -np.sin(motion_blur_angle), 0.],
        [np.sin(motion_blur_angle),  np.cos(motion_blur_angle), 0.],
        [0.,                         0.,                        1.],
    ])
    total_transformation = np.matmul(translation,
                                     np.matmul(np.matmul(rotation_matrix, scale_matrix),
                                               inverse_translation))

    kernel = cv2.warpAffine(kernel, total_transformation[:2, :], (kernel.shape[1], kernel.shape[0]))

    blurred = signal.convolve2d(target_image[:, :, 0], kernel)
    return blurred, kernel


def _get_gaussian_kernel(size, sigma):
    ax = np.linspace(-(size - 1.)/2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sigma))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)


def _random_perspective_distortion():
    g, = np.random.normal(-0.002, 0.001, 1)
    h, = np.random.normal(0., 0.0015, 1)

    matrix = np.array([
        [1., 0., 0.],
        [0., 1., 0.],
        [h,  g,  1.],
    ])

    return matrix


def _random_rotation_distortion():
    random_sample, = np.random.normal(0., 1./3., 1)
    angle = (random_sample * MAX_ROTATION_ANGLE_DEG * np.pi) / 180.

    matrix = np.array([
        [np.cos(angle), -np.sin(angle), 0.],
        [np.sin(angle), np.cos(angle),  0.],
        [0.,            0.,             1.],
    ])

    return matrix


def _random_scale_transformation():
    scale_factor, = np.random.uniform(0.05, 0.5, 1)
    matrix = np.array([
        [scale_factor, 0.,           0.],
        [0.,           scale_factor, 0.],
        [0.,           0.,           1.],
    ])

    return matrix


def _random_distortion(target_image):
    h, w, _ = target_image.shape

    translation = np.array([
        [1., 0., h/2],
        [0., 1., w/2],
        [0., 0., 1.],
    ])

    inverse_translation = np.array([
        [1., 0., -h/2],
        [0., 1., -w/2],
        [0., 0.,  1.],
    ])

    perspective_matrix = _random_perspective_distortion()
    rotation_matrix = _random_rotation_distortion()
    scale_matrix = _random_scale_transformation()

    transformation_matrix = np.matmul(perspective_matrix, rotation_matrix)
    transformation_matrix = np.matmul(transformation_matrix, scale_matrix)

    total_transformation = np.matmul(
        np.matmul(translation, transformation_matrix), inverse_translation)
    total_transformation = total_transformation / total_transformation[2, 2]

    warped = cv2.warpPerspective(target_image, total_transformation, target_image.shape[:2])

    return warped


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
