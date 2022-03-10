import torch.nn as nn
import torch.optim as optim
import json
from dataloader import WiCDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load Data
# Training
with open('padded_sequences.json') as q:
    ps_dict = json.load(q)

# Dev
with open('padded_sequences_dev.json') as f:
    ps_dict_dev = json.load(f)

train_data = []
dev_data = []

for i in range(len(ps_dict['labels'])):
    train_data.append(
        [ps_dict['sentence1s'][i], ps_dict['sentence2s'][i], ps_dict['targets'][i], ps_dict['labels'][i]])

for i in range(len(ps_dict_dev['labels'])):
    dev_data.append(
        [ps_dict_dev['sentence1s'][i], ps_dict_dev['sentence2s'][i], ps_dict_dev['targets'][i],
         ps_dict_dev['labels'][i]])

train_loader = DataLoader(
    WiCDataset(train_data),
    batch_size=32,
    shuffle=True,
)

dev_loader = DataLoader(
    WiCDataset(dev_data),
    batch_size=32,
    shuffle=True,
)

# Load embeddings matrix
emb_matrix = torch.load('embedding_matrix_new.pt')  # ->  torch.load('embeddings.pt')
vocab_size = emb_matrix.shape[0]  # 26684
embedding_dimension = emb_matrix.shape[1]  # 300

emb_matrix = torch.tensor(emb_matrix)


class MultilayerPerceptron(nn.Module):
    def __init__(self, hidden_size, target_weighting=0.005):
        super(MultilayerPerceptron, self).__init__()
        self.target_weighting = target_weighting
        self.hidden_size = hidden_size
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.embedding = nn.Embedding(vocab_size, embedding_dimension, padding_idx=0)
        self.embedding = self.embedding.from_pretrained(emb_matrix)
        self.dropout = nn.Dropout(0.9)
        # Multiplying embedding_dimension with two because of concatenation
        self.fc1 = nn.Linear(embedding_dimension * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 2)

    def forward(self, s1, s2, t1, training=False):
        target1 = t1[:, 0]
        target2 = t1[:, 1]
        # print(target1)
        # print(target2)
        # exit()

        e1 = self.embedding(s1)
        e2 = self.embedding(s2)

        target1_embeddings = e1[range(e1.shape[0]), target1.long()]
        target2_embeddings = e2[range(e2.shape[0]), target2.long()]

        e1 = torch.mean(e1, dim=1)  # [32,300] bs, emb. dim
        e2 = torch.mean(e2, dim=1)

        comb1 = e1 + target1_embeddings * self.target_weighting
        comb2 = e2 + target2_embeddings * self.target_weighting

        concat_emb = torch.cat((comb1, comb2), dim=-1).float()  # [32,600]

        h = self.fc1(concat_emb)
        h = F.relu(h)

        if training:
            h = self.dropout(h)
        h = F.relu(self.fc2(h))
        if training:
            h = self.dropout(h)
        h = self.fc3(h)
        return h


device = "cuda" if torch.cuda.is_available() else "cpu"

lr = 1e-4
model = MultilayerPerceptron(hidden_size=512)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

loss_fn = nn.CrossEntropyLoss()

epoch = 1000

import matplotlib.pyplot as plt

model_path = 'models/modelmodel7'
plot_path = 'plots/yesyes7.png'

if __name__ == '__main__':
    val_losses = []
    train_losses = []

    val_accuracy = []
    train_accuracy = []

    for i in range(epoch):
        epoch_loss = 0
        epoch_acc = 0
        dev_epoch_loss = 0
        dev_epoch_acc = 0
        num_iterations = 0

        for s1, s2, ts, ls in train_loader:
            sentence1 = torch.stack(s1).T.to(device).long()
            sentence2 = torch.stack(s2).T.to(device).long()
            targets = torch.stack(ts).T.to(device).float()
            labels = torch.stack(ls).T.to(device).long()

            print(sentence1)
            print(sentence1.shape)

            print(labels)
            print(labels.shape)
            exit()

            logits = model(sentence1, sentence2, targets, training=True)
            optimizer.zero_grad()
            loss = loss_fn(logits, labels.squeeze())

            train_losses.append(loss.item())

            loss.backward()
            # Gradient clipping
            optimizer.step()

            prediction = F.softmax(logits, dim=-1)
            prediction = torch.argmax(prediction, dim=-1)

            accuracy = torch.mean((prediction == labels.squeeze()).float())

            epoch_acc += accuracy.item()
            epoch_loss += loss.item()
            num_iterations += 1

        # since we're not training, we don't need to calculate the gradients for our outputs

        train_accuracy.append(epoch_acc / num_iterations)

        print("train_acc: :", epoch_acc / num_iterations)
        with torch.no_grad():

            num_iterations = 0
            for sd1, sd2, tds, lds in dev_loader:
                sentence1_dev = torch.stack(sd1).T.to(device).long()
                sentence2_dev = torch.stack(sd2).T.to(device).long()
                targets_dev = torch.stack(tds).T.to(device).float()
                labels_dev = torch.stack(lds).T.to(device).long()

                dev_logits = model(sentence1_dev, sentence2_dev, targets_dev)
                # the class with the highest energy is what we choose as prediction

                dev_loss = loss_fn(dev_logits, labels_dev.squeeze())

                val_losses.append(dev_loss.item())
                dev_prediction = F.softmax(dev_logits, dim=-1)
                dev_prediction = torch.argmax(dev_prediction, dim=-1)

                dev_accuracy = torch.mean((dev_prediction == labels_dev.squeeze()).float())

                dev_epoch_acc += dev_accuracy.item()
                dev_epoch_loss += dev_loss.item()
                num_iterations += 1

        val_accuracy.append(dev_epoch_acc / num_iterations)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': dev_epoch_loss / num_iterations,
        }, model_path)
        print("val_acc:", dev_epoch_acc / num_iterations)

    plt.figure(figsize=(10, 5))
    plt.title("Training and Validation Accuracy")
    plt.plot(val_accuracy, label="val")
    plt.plot(train_accuracy, label="train")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig(plot_path)
    plt.legend()
    plt.show()
