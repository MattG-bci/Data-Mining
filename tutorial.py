import torch
import clip
from PIL import Image
import numpy as np

model, preprocess = clip.load("RN50")

image = preprocess(Image.open("test_image.jpg")).unsqueeze(0)
n_labels = int(input("How many labels you want to consider: "))

labels = [input("Enter your caption: ") for _ in range(n_labels)]
text = clip.tokenize(labels)

with torch.no_grad():
	img_features = model.encode_image(image)
	text_features = model.encode_text(text)
	logits_img, logits_text = model(image, text)
	probs = logits_img.softmax(dim=-1).numpy()

print(f"Label probabilities: {probs}")
print(f"Your image is representing: {labels[np.argmax(probs)]}")

