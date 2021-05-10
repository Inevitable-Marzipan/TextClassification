import numpy as np

from utils import load_data
from utils import evaluate
from nlptools.stringprocessing import process_texts

from nlptools.model import NBTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Load in data
data_folder = 'imdb'
train, dev = load_data(data_folder)
train_text, train_labels = zip(*train)
dev_text, dev_labels = zip(*dev)

# Process Text
train_processed = process_texts(train_text)
train_input = [" ".join(s) for s in train_processed]
dev_processed = process_texts(dev_text)
dev_input = [" ".join(s) for s in dev_processed]

# Convert Labels
train_labels = np.array(train_labels)
dev_labels = np.array(dev_labels)

# Define Model
tfidf = TfidfVectorizer(ngram_range=(1, 1), min_df=3, max_df=1.)
clf = LogisticRegression(solver='lbfgs', C=1., random_state=42, n_jobs=-1)
nbt = NBTransformer(alpha=1)
pipe = Pipeline([('tfidf', tfidf),
                ('nbt', nbt),
                ('clf', clf)])

# Fit Model
pipe.fit(train_input, train_labels)

# Evaluate
train_predictions = pipe.predict(train_input)
dev_predictions = pipe.predict(dev_input)

print(evaluate(train_labels, train_predictions))
print(evaluate(dev_labels, dev_predictions))