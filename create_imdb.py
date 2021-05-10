import os
import csv
import random
import thinc.extra.datasets

random.seed(42)

def load_data(limit=0, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    train_data, _ = thinc.extra.datasets.imdb()
    random.shuffle(train_data)
    train_data = train_data[-limit:]
    texts, labels = zip(*train_data)
    split = int(len(train_data) * split)
    return (texts[:split], labels[:split]), (texts[split:], labels[split:])

# Load Data
n_texts = 2000
(train_texts, train_labels), (dev_texts, dev_labels) = load_data()
train_texts = train_texts[:n_texts]
train_labels = train_labels[:n_texts]

# Save Data
data_dir = 'imdb'
os.makedirs(data_dir, exist_ok=True)


with open(os.path.join(data_dir, 'train.csv'), 'w', encoding='utf-8', newline='') as f:
    write = csv.writer(f) 
    write.writerows(zip(train_texts, train_labels))

with open(os.path.join(data_dir, 'dev.csv'), 'w', encoding='utf-8', newline='') as f:
    write = csv.writer(f) 
    write.writerows(zip(dev_texts, dev_labels))