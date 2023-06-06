import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def visualise_top_images(images, scores):
    for image, score in zip(images, scores):
        image = np.array(Image.open(image))
        plt.imshow(image)
        plt.title(f"Matching score: {score:.4f}")
        plt.show()
