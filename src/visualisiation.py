import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import matplotlib.gridspec as gridspec

def visualise_top_images(images, scores):
    n_img = len(images)
    n_cols = math.ceil(n_img / 3)
    plt.rcParams["figure.figsize"] = [12, 12]
    plt.rcParams["figure.autolayout"] = True
    gs1 = gridspec.GridSpec(3, n_cols)
    gs1.update(wspace=0.1, hspace=0.1)
    plt.title(f"Top {n_img} matches", fontsize=10)
    plt.tight_layout()
    for idx, (image, score) in enumerate(zip(images, scores)):
        ax1 = plt.subplot(gs1[idx])
        image = np.array(Image.open(image))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Rank {idx + 1}", fontsize=20)
    plt.savefig("pic.jpg")


