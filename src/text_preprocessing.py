import re
import string
from nltk.corpus import stopwords

def process_caption(caption):
    stopwords_list = stopwords.words("english")
    caption = caption.lower() # lowercasing all the words
    caption = "".join(char for char in caption if char not in string.punctuation) # removing redundant characters
    caption = " ".join(word for word in caption.split() if (word not in stopwords_list) and len(word) > 2) # removing stopwords
    return caption

if __name__ == "__main__":
    caption = input("Enter your caption: ")
    processed_caption = process_caption(caption)
    print(f"Pre processing caption: {caption}")
    print(f"Caption after the processing: {processed_caption}")
