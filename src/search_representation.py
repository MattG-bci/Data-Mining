import torch
import clip
from PIL import Image
import numpy as np


def clip_representations(images, caption):
	model, preprocess = clip.load("RN50")
	images_preprocessed = [preprocess(Image.open(image)).unsqueeze(0) for image in images]
	text = clip.tokenize(caption)

	with torch.no_grad():
		logits = []
		for img in images_preprocessed:
			logits_img, logits_text = model(img, text)
			logits.append(logits_img)
		logits = torch.tensor(logits)
		probs = logits.softmax(dim=-1).numpy()
	return probs

