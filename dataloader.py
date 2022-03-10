import torch
from Vocabulary import Vocabulary
import json
import ast
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torch.nn.utils.rnn import pad_sequence

with open('dev_sentences.txt') as f:
    s = f.readlines()
s = [x.strip() for x in s]

with open('dev_targets.txt') as f:
    t = f.readlines()
t = [x.strip() for x in t]

with open('dev_labels.txt') as f:
    l = f.readlines()
l = [x.strip() for x in l]

with open('sentences.txt') as f:
    s_target = f.readlines()
s_target = [x.strip() for x in s_target]

target_sentences_as_list = []
sentences_as_list = []
targets_as_list = []
labels_as_list = []

for s_to_list in s:
    s_to_list = ast.literal_eval(s_to_list)
    sentences_as_list.append(s_to_list)

for t_to_list in t:
    t_to_list = ast.literal_eval(t_to_list)
    targets_as_list.append(t_to_list)

for l_to_list in l:
    l_to_list = ast.literal_eval(l_to_list)
    labels_as_list.append(l_to_list)

for ts_to_list in s_target:
    ts_to_list = ast.literal_eval(ts_to_list)
    target_sentences_as_list.append(ts_to_list)


l = torch.tensor(labels_as_list).int().tolist()


# This is for cleaning....
class NLP_Dataset(Dataset):

    def __init__(self, sentences: list, targets: list, labels: list):

        self.PAD_token = "[pad]"
        self.SEP_token = "[sep]"
        self.UNK_token = "[unk]"
        self.num_words = 3
        self.sentences = sentences
        self.targets = targets
        self.labels = labels
        self.word2index = {self.PAD_token: 0, self.SEP_token: 1, self.UNK_token: 2}
        self.word2count = {self.PAD_token: 0, self.SEP_token: 0, self.UNK_token: 0}
        self.index2word = {self.PAD_token: "[pad]", self.SEP_token: "[sep]", self.UNK_token: "[unk]"}
        self.longest_sentence = 0
        self.padded_sequences = []

    def build_voc(self, sentences):
        for sentence_list in sentences:

            max_len = max(len(sentence_list[0]), len(sentence_list[1]))

            if max_len > self.longest_sentence:
                self.longest_sentence = max_len

            for word in sentence_list[0]:

                if word not in self.word2index:
                    self.word2index[word] = self.num_words
                    self.word2count[word] = 1
                    self.index2word[self.num_words] = word
                    self.num_words += 1
                else:
                    self.word2count[word] += 1

            for word in sentence_list[1]:

                if word not in self.word2index:
                    self.word2index[word] = self.num_words
                    self.word2count[word] = 1
                    self.index2word[self.num_words] = word
                    self.num_words += 1
                else:
                    self.word2count[word] += 1

    def to_word(self, index):
        return self.index2word[index]

    def to_index(self, word):
        return self.word2index[word]

    # Use once
    def dump_all_to_json(self):
        dict = {
            'w2i': self.word2index,
            'i2w': self.index2word,
            'word_counts': self.word2count,
            'longest': self.longest_sentence,
        }

        with open('vocab_dictionary.json', 'w') as fp:
            json.dump(dict, fp)

    def read_json(self, path='vocab_dictionary.json'):
        with open(path) as f:
            data = json.load(f)
        return data

    # Also do the padding here
    def convert_each_sent_to_index(self):
        comb1 = []
        comb2 = []

        for sentences in sentences_as_list:
            sentence1_index = []
            sentence2_index = []
            for word in sentences[0]:
                try:
                    sentence1_index.append(self.to_index(word))
                except KeyError:
                    sentence1_index.append(self.to_index(self.UNK_token))
            for wordx in sentences[1]:
                try:
                   sentence2_index.append(self.to_index(wordx))
                except KeyError:
                    sentence2_index.append(self.to_index(self.UNK_token))


            comb1.append(sentence1_index)
            comb2.append(sentence2_index)

        # Padding here
        for i in comb1:
            if len(i) < self.longest_sentence:
                for x in range(0, self.longest_sentence - len(i)):
                    i.append(self.to_index(self.PAD_token))
        for ix in comb2:
            if len(ix) < self.longest_sentence:
                for xx in range(0, self.longest_sentence - len(ix)):
                    ix.append(self.to_index(self.PAD_token))

            # pad_array = [0] * (self.longest_sentence - len(i))
            # i+= pad_array
        return comb1, comb2

    # Use once
    def dump_padseqs_to_json(self):
        sentence1s, sentence2s = self.convert_each_sent_to_index()

        dict = {
            'sentence1s': sentence1s,
            'sentence2s': sentence2s,
            'targets': targets_as_list,
            'labels': l
        }

        with open('padded_sequences_dev.json', 'w') as fp:
            json.dump(dict, fp)


d = NLP_Dataset(sentences_as_list, t, l)
d.build_voc(target_sentences_as_list)
d.convert_each_sent_to_index()
d.dump_padseqs_to_json()



# Get padded sequences
# dict: {sentence1s:....., sentence2s:.....}
# ps_dict = d.read_json(path='padded_sequences_dev.json')



# print(ps_dict)


# Create training data
# train_data = []
# for i in range(len(ps_dict['labels'])):
#     train_data.append([ps_dict['sentence1s'][i], ps_dict['sentence2s'][i], ps_dict['targets'][i], ps_dict['labels'][i]])
#




#
# class MyDataLoader:
#     def __init__(self, train_data, batch_size, shuffle=False):
#         self.sentence1s = [data[0] for data in train_data]
#         self.sentence2s = [data[1] for data in train_data]
#         self.targets = [data[2] for data in train_data]
#         self.labels = [data[3] for data in train_data]
#         self.total_length = len(train_data)
#         self.c = 0
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#
#     def __len__(self):
#         return self.total_length
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#
#         if self.c < self.total_length:
#             last = min(self.c + self.batch_size, self.total_length)
#             s1_batch = torch.tensor(self.sentence1s[self.c:last])
#             s2_batch = torch.tensor(self.sentence2s[self.c:last])
#             targets_batch = torch.tensor(self.targets[self.c:last])
#             labels_batch = torch.tensor(self.labels[self.c:last])
#             self.c += self.batch_size
#             return s1_batch, s2_batch, targets_batch, labels_batch
#         else:
#             self.c = 0
#             raise StopIteration()
#

class WiCDataset(Dataset):
    def __init__(self, train_data):
        self.sentence1s = [data[0] for data in train_data]
        self.sentence2s = [data[1] for data in train_data]
        self.targets = [data[2] for data in train_data]
        self.labels = [data[3] for data in train_data]
        self.total_length = len(train_data)

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):

        return self.sentence1s[index],self.sentence2s[index],self.targets[index], [self.labels[index]]



# hd = MyDataLoader(train_data, 32, shuffle=False)
# b1 = next(iter(hd))
# print(b1[3][0:10])




# -----    MyDataLoader returns    -----
# Sentences1 -> instance[0] torch.Size([32, 48])
# Sentences2 -> instance[1] torch.Size([32, 48])
# Targets ->    instance[2] torch.Size([32, 2])  Including target1 and target2
# Labels ->     instance[3] torch.Size([32])

# Create instances to feed to the model
# a = torch.tensor(ps_dict['sentence1s'])
# print(a[0])
# target = torch.tensor(ps_dict['targets'])
# print(target[0])
# label = torch.tensor(ps_dict['labels'])
# print(label[0])
