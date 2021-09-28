import numpy as np
import matplotlib.pyplot as plt
import cv2


def main():
    image = cv2.imread("motion_blur.jpg")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plot_harris(gray)


def create_checker_corner(image, y, x, h, w):
    image[y:y + h//2, x:x + w//2] = 1.
    image[y + h//2:y + h, x + w//2:x + w] = 1.

def create_checker_target():
    width = 200
    image = np.zeros((width, width), dtype=np.float32)

    corner_width = 100

    corner_points = [
        (0, 0),
        (width - corner_width, 0),
        (0, width - corner_width),
        (width - corner_width, width - corner_width),
    ]

    for y, x in corner_points:
        create_checker_corner(image, y, x, corner_width, corner_width)

    cv2.imwrite("checker.png", (image * 255).astype(np.uint8))

    return image

def plot_harris(image):
    harris_image = cv2.cornerHarris(image, 3, 3, 0.04)

    plt.figure(1)
    plt.subplot(1, 2, 1)
    plt.imshow(image)

    plt.subplot(1, 2, 2)
    plt.imshow(harris_image)

    plt.show()


if __name__ == "__main__":
    main()
