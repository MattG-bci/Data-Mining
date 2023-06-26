import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import matplotlib.gridspec as gridspec

def visualise_top_images(images, scores):
    n_img = len(images)
    n_rows = n_img // 2
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["figure.autolayout"] = False
    gs1 = gridspec.GridSpec(n_rows, 2)
    plt.title(f"Top {n_img} matches", fontsize=10)
    plt.tight_layout()
    for idx, (image, score) in enumerate(zip(images, scores)):
        ax1 = plt.subplot(gs1[idx])
        image = np.array(Image.open(image))
        plt.imshow(image)
        plt.axis("off")
        plt.title(f"Rank {idx + 1} / Score: {score:.3f}", fontsize=20)
        plt.gca().set_aspect("auto")
    plt.subplots_adjust(wspace=0.1, hspace=0.2)
    plt.savefig("pic.jpg")


def plot_histogram(d):
    # An "interface" to matplotlib.axes.Axes.hist() method
    n, bins, patches = plt.hist(x=d, bins='auto', color='#0504aa',
                            alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('My Very Own Histogram')
    plt.text(23, 45, r'$\mu=15, b=3$')
    maxfreq = n.max()
# Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)
    plt.savefig("histogram.jpg")
