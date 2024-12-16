import torch
from transformers import DistilBertTokenizer, DistilBertModel
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def load_data(data):
    text_data = [entry['sentence_normalized'] for entry in data]
    labels = [entry['polarity'] for entry in data]
    return text_data, labels

def train_distilbert(text_data, labels):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)


    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __getitem__(self, idx):
            input_ids = self.inputs['input_ids'][idx].to(torch.long)
            attention_mask = self.inputs['attention_mask'][idx].to(torch.long)
            labels = torch.tensor(self.labels[idx])
            return input_ids, attention_mask, labels

        def __len__(self):
            return len(self.labels)

    dataset = SentimentDataset(inputs, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

    class SentimentClassifier(nn.Module):
        def __init__(self):
            super(SentimentClassifier, self).__init__()
            self.dropout = nn.Dropout(0.1)
            self.classifier = nn.Linear(768, 3)

        def forward(self, outputs):
            pooled_output = outputs.last_hidden_state[:, 0, :]
            pooled_output = self.dropout(pooled_output)
            outputs = self.classifier(pooled_output)
            return outputs

    sentiment_classifier = SentimentClassifier()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device:", device)
    model.to(device)
    sentiment_classifier.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(sentiment_classifier.parameters()), lr=1e-5)

    for epoch in range(5):
        model.train()
        sentiment_classifier.train()
        total_loss = 0
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask)
            outputs = sentiment_classifier(outputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}, Loss: {total_loss / len(data_loader)}')

    return model, sentiment_classifier, device

def evaluate_distilbert(model, sentiment_classifier, device, text_data, labels):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    inputs = tokenizer(text_data, return_tensors='pt', padding=True, truncation=True)

    class SentimentDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, labels):
            self.inputs = inputs
            self.labels = labels

        def __getitem__(self, idx):
            input_ids = self.inputs['input_ids'][idx]
            attention_mask = self.inputs['attention_mask'][idx]
            labels = self.labels[idx]
            return input_ids, attention_mask, labels

        def __len__(self):
            return len(self.labels)

    dataset = SentimentDataset(inputs, labels)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    sentiment_classifier.eval()
    total_correct = 0
    predicted_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            outputs = sentiment_classifier(outputs)
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            predicted_labels.extend(predicted.cpu().numpy())

    accuracy = total_correct / len(dataset)
    roc_auc = roc_auc_score(labels, predicted_labels, multi_class='ovr')
    report = classification_report(labels, predicted_labels)

    return accuracy, roc_auc, report