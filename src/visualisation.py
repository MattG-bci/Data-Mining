import os

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import matplotlib.gridspec as gridspec

def visualise_top_images(images, scores, caption):
    n_img = len(images)
    if n_img % 2 == 0:
        n_rows = n_img // 2
    else:
        n_rows = n_img // 2 + 1

    plt.rcParams["figure.figsize"] = [32 + n_img, 20 + 3 * (n_img // 1)]
    plt.rcParams["figure.autolayout"] = False
    gs1 = gridspec.GridSpec(n_rows, 2)
    plt.title(f"Top {n_img} matches", fontsize=20 + 2*int(n_img))
    plt.tight_layout()
    for idx, (image, score) in enumerate(zip(images, scores)):
        ax1 = plt.subplot(gs1[idx])
        image = np.array(Image.open(image))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Rank {idx + 1}", fontsize=30 + 2 * int(n_img))
        plt.gca().set_aspect("auto")
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    out_path = "./output visualisations"
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    plt.savefig(f"{out_path}/{caption}.jpg")
    plt.close()

