import torch
import clip
from PIL import Image
import numpy as np
import time
from torchvision import transforms


def latency_wrapper(func):
	def timeit(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f"The function {func.__name__} has executed in {(end - start):.4f} seconds")
		return result
	return timeit


def search_model(backbone, device):
	model, preprocess = clip.load(backbone, device=device)
	return model, preprocess


@latency_wrapper
def search_frames(images, caption, model):
	device = "cuda" if torch.cuda.is_available() else "cpu"
	text = clip.tokenize(caption).to(device)
	with torch.no_grad():
		images = images.to(device)
		logits_img, logits_text = model(images, text)
		images.detach()
		text.detach()

	return logits_img

