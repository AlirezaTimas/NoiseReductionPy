import cv2
import numpy as np

def addingspecklenoise(image, mean=0, sigma=0.1):
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_image = image + image * noise
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def convolutionfilter(image, kernel):
    return cv2.filter2D(image, -1, kernel)

image = cv2.imread(r"D:\Pictures\images.jpg", cv2.IMREAD_GRAYSCALE)

noisy_image = addingspecklenoise(image)

kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]], dtype=np.float32)
kernel = kernel / np.sum(kernel)


denoised_image = convolutionfilter(noisy_image, kernel)

cv2.imshow("OriginalImage", image)
cv2.imshow("NoisyImage", noisy_image)
cv2.imshow("DenoisedImage", denoised_image)

cv2.waitKey(0)