import numpy as np
import pandas as pd
import json
import torch

with open('vocab_dictionary.json') as f:
    vocab_file = json.load(f)

wi = vocab_file['w2i']


glove = pd.read_csv('embeddings/glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)

glove_embedding = {key: val.values for key, val in glove.T.items()}


def create_embedding_matrix(word_index, embedding_dict):
    embedding_matrix = np.zeros((len(word_index) + 1, 300))

    for word, index in word_index.items():
        if word in embedding_dict:
            embedding_matrix[index] = embedding_dict[word]
    return embedding_matrix

glove_ready = create_embedding_matrix(wi,glove_embedding)

# Adding average for [unk] token ( index 2 )
avg = 0
sum = 0
cnt = 0
for i in glove_ready:
    sum += i
    cnt +=1

avg = sum/cnt

glove_ready[2] = avg


torch.save(glove_ready,'embedding_matrix_new.pt')