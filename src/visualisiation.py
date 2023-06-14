import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math

def visualise_top_images(images, scores):
    n_img = len(images)
    n_cols = math.ceil(n_img / 3)
    plt.figure(figsize=(12, 12))
    for idx, (image, score) in enumerate(zip(images, scores)):
        plt.subplot(n_cols, 3, idx + 1)
        image = np.array(Image.open(image))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Rank {idx + 1}", fontsize=12)
    plt.savefig("pic.jpg")


