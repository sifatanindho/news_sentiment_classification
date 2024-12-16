import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def load_data(data):
    text_data = [entry['sentence_normalized'] for entry in data]
    labels = [entry['polarity'] for entry in data]
    return text_data, labels

def train_naive_bayes(text_data, labels, alpha=1.0, fit_prior=True):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(text_data)
    y = np.array(labels)

    # Map the labels to integers (0, 1, 2)
    label_map = {2: 0, 4: 1, 6: 2}
    y = np.array([label_map[label] for label in y])
    clf = MultinomialNB(alpha=alpha, fit_prior=fit_prior)
    clf.fit(X, y)
    return clf, vectorizer

def evaluate_naive_bayes(clf, vectorizer, text_data, labels):
    X = vectorizer.transform(text_data)
    y = np.array(labels)

    # Map the labels to integers (0, 1, 2)
    label_map = {2: 0, 4: 1, 6: 2}
    y = np.array([label_map[label] for label in y])
    
    predictions = clf.predict(X)
    predictions_proba = clf.predict_proba(X)

    accuracy = accuracy_score(y, predictions)
    print('Classification Report:')
    roc_auc = roc_auc_score(y, predictions_proba, average='macro', multi_class='ovr')
    report = classification_report(y, predictions, digits=4, output_dict=True)
    print(report)
    print(f'Accuracy: {accuracy:.4f}')
    print(f'ROC-AUC: {roc_auc:.4f}')

    return accuracy, roc_auc, report