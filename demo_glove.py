import time

import numpy as np
import pandas as pd
import csv

# Get embeddings from here https://nlp.stanford.edu/projects/glove/
# print("Loading embeddings...")
# start = time.time()
# data = pd.read_csv(
#     'glove.840B.300d.txt',
#     sep=" ",
#     index_col=0,
#     header=None,
#     quoting=csv.QUOTE_NONE,
#     na_values=None,
#     encoding='utf-8',
#     keep_default_na=False
# )
# print(f"Done in {time.time() - start} seconds.")  # Done in 59.05877685546875 seconds.
# print("Processing embeddings...")
# start = time.time()
# embeddings_dict = {key: val.values for key, val in data.T.items()}
# del data
# print(f"Done in {time.time() - start} seconds.")  # Done in 94.05732679367065 seconds.

# print("Loading data...")
# start = time.time()
# glove = np.loadtxt(
#     'glove.840B.300d.txt',
#     dtype='str',
#     comments=None,
#     delimiter=' ',
# )
# words = glove[:, 0]
# vectors = glove[:, 1:].astype('float')
# print(f"Done in {time.time() - start} seconds.")  # More than I had patience for.

print("Loading data...")
start = time.time()
embeddings_dict = {}
with open('glove.840B.300d.txt', 'r', encoding='utf-8') as f:
    for line in f:
        split_line = line.split(" ")
        word = split_line[0]
        embedding = np.array(split_line[1:], dtype=np.float64)
        embeddings_dict[word] = embedding
print(f"Done in {time.time() - start} seconds.")  # Done in 78.3701286315918 seconds.


def get_similar(word):
    try:
        emb = embeddings_dict[word.lower()]
        # Get normalised embedding
        norm_emb = np.linalg.norm(emb)
        # Get similarities
        similarity = []
        for word in embeddings_dict:
            sim = np.dot(emb, embeddings_dict[word]) / (norm_emb * np.linalg.norm(embeddings_dict[word]))
            similarity.append((word, sim))

        # Get most similar
        similarity.sort(key=lambda x: x[1], reverse=True)
        return similarity[:10]
    except KeyError:
        return None


print(get_similar("cat"))
print(get_similar("king"))
print(get_similar("test"))
