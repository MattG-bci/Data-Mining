import torch
import clip
from PIL import Image
import numpy as np
import time


def latency_wrapper(func):
	def timeit(*args, **kwargs):
		start = time.time()
		result = func(*args, **kwargs)
		end = time.time()
		print(f"The function {func.__name__} has executed in {(end - start):.4f} seconds")
		return result
	return timeit



@latency_wrapper
def search_frames(images, caption):
	model, preprocess = clip.load("RN101")
	images_preprocessed = [preprocess(Image.open(image)).unsqueeze(0) for image in images]
	text = clip.tokenize(caption)

	with torch.no_grad():
		logits = []
		for img in images_preprocessed:
			logits_img, logits_text = model(img, text)
			logits.append(logits_img)
		logits = torch.tensor(logits)
		probs = logits.softmax(dim=-1).numpy()
	return logits

