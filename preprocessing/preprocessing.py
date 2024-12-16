import json
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.stem import PorterStemmer
import string
import spacy

import matplotlib.pyplot as plt
import math
from collections import Counter

# Load the dataset
class Preprocessing:
    
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def load_dataset(file_path):
        data = []
        with open(file_path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    
    # Splitting the dataset into target categories
    def split_data_by_target(data):
        polarity_data = {2.0: [], 4.0: [], 6.0: []}

        for entry in data:
            for target in entry['targets']:
                polarity = target['polarity']
                polarity_data[polarity].append(entry)

        return polarity_data

    # Distribution of sentiment labels
    def find_distribution(data):
        polarities = [target['polarity'] for entry in data for target in entry['targets']]
        print("Sentiment Class Distribution:")
        print(Counter(polarities))
        plt.bar(Counter(polarities).keys(), Counter(polarities).values())
        plt.xlabel('Sentiment Class')
        plt.ylabel('Frequency')
        plt.show()

    # Sentence Length Distribution
    def find_sentence_lengths(data):
        sentence_lengths = [len(entry['sentence_normalized'].split()) for entry in data]
        print("\nSentence Length Distribution:")
        print(Counter(sentence_lengths))

        # Histogram
        plt.hist(sentence_lengths, bins=20)
        plt.xlabel('Sentence Length')
        plt.ylabel('Frequency')
        plt.show()

        # Boxplot
        plt.boxplot(sentence_lengths)
        plt.ylabel('Sentence Length')
        plt.show()

        # Pie chart 
        bins = [0, 20, 50, 100, 200, float('inf')]
        labels = ['0-20', '20-50', '50-100', '100-200', '200+']

        counts = [sum(1 for length in sentence_lengths if low <= length < high) 
                  for low, high in zip(bins, bins[1:])]

        plt.figure(figsize=(8, 6))
        wedges, _, autotexts = plt.pie(counts, labels=["" for _ in labels], autopct='%1.1f%%', startangle=90)
        plt.legend(wedges, labels, title="Length Ranges", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

# Add title
        # for wedge, count in zip(wedges, counts):
        #     angle = (wedge.theta2 + wedge.theta1) / 2  # Get the angle of the wedge
        #     x = 1.1 * math.cos(math.radians(angle))  # X position for the arrow
        #     y = 1.1 * math.sin(math.radians(angle))  # Y position for the arrow
        #     plt.annotate(f'{count / sum(counts) * 100:.1f}%', xy=(x, y), xytext=(1.3 * x, 1.3 * y), 
        #                 arrowprops=dict(facecolor='black', arrowstyle='->'), 
        #                 ha='center', va='center')

        plt.title('Sentence Length Distribution')
        plt.show()

    # High-frequency words
    def find_high_frequency_words(data):
        words = [word for entry in data for word in entry['sentence_normalized'].split()]
        print("\nHigh-Frequency Words:")
        print(Counter(words).most_common(5))
        plt.bar([word[0] for word in Counter(words).most_common(5)], [word[1] for word in Counter(words).most_common(5)], width=0.8)
        plt.xlabel('Word')
        plt.ylabel('Frequency')
        plt.show()

    # Remove stopwords
    def remove_stopwords(data):
        stop_words = set(stopwords.words('english'))
        for entry in data:
            entry['sentence_normalized'] = ' '.join([word for word in entry['sentence_normalized'].split() 
                                                     if word not in stop_words])
        return data
    
    # Find co-occuring words
    def calculate_co_occurrence(data, n=2):
        co_occurrence = []
        nltk.download('punkt_tab')
        for entry in data:
            sentence = entry['sentence_normalized']
            tokens = word_tokenize(sentence)
            bigrams = list(ngrams(tokens, n))
            co_occurrence.extend(bigrams)

        co_occurrence_freq = Counter(co_occurrence)

        top_10_bigrams = co_occurrence_freq.most_common(10)
        bigrams, freqs = zip(*top_10_bigrams)

        # Convert bigrams to strings for plotting
        bigram_labels = [' '.join(bigram) for bigram in bigrams]

        plt.figure(figsize=(10, 6))
        plt.bar(bigram_labels, freqs)  # Use bigram_labels instead of bigrams
        plt.title('Top 10 Most Common Bigrams')
        plt.xlabel('Bigram')
        plt.ylabel('Frequency')
        plt.xticks(rotation=90)
        plt.show()

        # Bigram frequency distribution
        # freqs = list(co_occurrence_freq.values())
        # plt.figure(figsize=(8, 6))
        # plt.hist(freqs, bins=10)
        # plt.title('Bigram Frequency Distribution')
        # plt.xlabel('Frequency')
        # plt.ylabel('Count')
        # plt.show()

    # Remove punctuation
    def remove_punctuation(data):
        translator = str.maketrans('', '', string.punctuation)
        for entry in data:
            entry['sentence_normalized'] = entry['sentence_normalized'].translate(translator)
        return data
    
    # Remove non-alphanumeric characters
    def remove_non_alphanumeric(data):
        for entry in data:
            entry['sentence_normalized'] = ''.join([c for c in entry['sentence_normalized'] if c.isalnum() or c.isspace()])
        return data
    
    # Porter Stemming
    def porter_stemming(data):
        stemmer = PorterStemmer()
        for entry in data:
            tokens = word_tokenize(entry['sentence_normalized'])
            stemmed_tokens = [stemmer.stem(token) for token in tokens]
            entry['sentence_normalized'] = ' '.join(stemmed_tokens)
        return data
    
    # Extract relevant columns
    def extract_relevant_columns(data):
        extracted_data = []
        for entry in data:
            targets = entry['targets']
            for target in targets:
                extracted_entry = {
                    'gid': entry['primary_gid'],
                    'sentence_normalized': entry['sentence_normalized'],
                    'polarity': target['polarity']
                }
                extracted_data.append(extracted_entry)
        return extracted_data
    
    # Write the preprocessed data to a file
    def write_preprocessed_data(data, destination):
        with open(destination, 'w') as f:
            for entry in data:
                json.dump(entry, f)
                f.write('\n')