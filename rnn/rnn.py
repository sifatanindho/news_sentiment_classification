import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class RNNClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(RNNClassifier, self).__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.rnn.num_layers, x.size(0), self.rnn.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out

class TextDataset(Dataset):
    def __init__(self, text_data, labels, vectorizer):
        self.text_data = text_data
        self.labels = labels
        self.vectorizer = vectorizer

    def __len__(self):
        return len(self.text_data)

    def __getitem__(self, idx):
        text = self.text_data[idx]
        label = self.labels[idx]

        X = self.vectorizer.transform([text]).toarray()
        y = label

        return torch.tensor(X, dtype=torch.float), torch.tensor(y, dtype=torch.long)

def load_data(data):
    text_data = [entry['sentence_normalized'] for entry in data]
    labels = [entry['polarity'] for entry in data]
    return text_data, labels

def train_rnn(text_data, labels, epochs, hidden_dim, num_layers, learning_rate):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)
    y = np.array(labels)
    label_map = {2: 0, 4: 1, 6: 2}
    y = np.array([label_map[label] for label in y])
    output_dim = len(np.unique(y))

    dataset = TextDataset(text_data, y, vectorizer)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    input_dim = X.shape[1]
    model = RNNClassifier(input_dim, hidden_dim, output_dim, num_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Device:', device)

    loss_history = []
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in data_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        loss_history.append(total_loss / len(data_loader))
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    plt.plot(range(1, epochs+1), loss_history)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.show()

    return model, vectorizer

def evaluate_rnn(model, vectorizer, text_data, labels):
    X = vectorizer.transform(text_data)
    y = np.array(labels)

    label_map = {2: 0, 4: 1, 6: 2}
    y = np.array([label_map[label] for label in y])

    dataset = TextDataset(text_data, y, vectorizer)
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    model.eval()
    total_correct = 0
    y_pred = []
    y_true = []
    y_pred_proba = []
    with torch.no_grad():
        for batch in data_loader:
            X_batch, y_batch = batch
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == y_batch).sum().item()
            y_pred.extend(predicted.cpu().numpy())
            y_true.extend(y_batch.cpu().numpy())
            y_pred_proba.extend(torch.softmax(outputs, dim=1).cpu().numpy())

    accuracy = total_correct / len(dataset)
    roc_auc = roc_auc_score(y_true, y_pred_proba, multi_class='ovr')
    print('Classification Report:')
    report = classification_report(y_true, y_pred, digits=4, output_dict=True)
    print(report)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')

    return accuracy, roc_auc, report