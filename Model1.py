import torch.nn as nn
import torch
from dataloader import NLP_Dataset
# with open('vocab_dictionary.json') as f:
#     vocab_file = json.load(f)
#
# # print(vocab_file['w2i'].items())
# # exit()
#
# # Take care of Glove embeddings first
# glove = pd.read_csv('embeddings/glove.6B.300d.txt', sep=" ", quoting=3, header=None, index_col=0)
# glove_embedding = {key: val.values for key, val in glove.T.items()}
#
#
# def create_emb_matrix(word2index, embedding_dict, dimension=300):
#     embedding_matrix = np.zeros((len(word2index.keys()) + 1, dimension))
#
#     for index, word in enumerate(word2index.keys()):
#         if word in embedding_dict:
#             embedding_matrix[index] = embedding_dict[word]
#     return embedding_matrix
#
#
# em = create_emb_matrix(vocab_file['w2i'], glove_embedding)


emb_matrix = torch.load('embeddings_matrix.pt') # ->  torch.load('embeddings.pt')


vocab_size = emb_matrix.shape[0]   # 26684
vector_size = emb_matrix.shape[1]  # 300

print(vocab_size)
print(vector_size)



# embeddings = nn.Embedding.from_pretrained(glove_embedding, freeze=True)

embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=vector_size)
embedding.weight = nn.Parameter(torch.tensor(emb_matrix, dtype=torch.float32))
print(embedding)