import random
import numpy as np
from utils import load_data
from utils import evaluate
import spacy
from spacy.util import minibatch, compounding

spacy.require_gpu()

# Load in data
data_folder = 'imdb'
train, dev = load_data(data_folder)
train_text, train_labels = zip(*train)
dev_text, dev_labels = zip(*dev)

train_cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in train_labels]
dev_cats = [{"POSITIVE": bool(y), "NEGATIVE": not bool(y)} for y in dev_labels]

# Convert Labels
train_labels = np.array(train_labels)
dev_labels = np.array(dev_labels)

# Train Data Format
train_data = list(zip(train_text, [{"cats": cats} for cats in train_cats]))

# SpaCy Model
#nlp = spacy.load('en_core_web_sm')
nlp = spacy.load('en_trf_distilbertbaseuncased_lg')
#nlp = spacy.blank('en')
textcat = nlp.create_pipe(
        'textcat',
        config={'exculsive_classes': True,
                'architecture': 'simple_cnn'}) # ensemble, bow, simple_cnn
nlp.add_pipe(textcat, last=True)

textcat.add_label('POSITIVE')
textcat.add_label('NEGATIVE')

# Train Model
n_iter = 10
# get names of other pipes to disable them during training
pipe_exceptions = ["textcat", "trf_wordpiecer", "trf_tok2vec"]
other_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]
#other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']
with nlp.disable_pipes(*other_pipes):  # only train textcat
    optimizer = nlp.begin_training()
    print("Training the model...")
    for i in range(n_iter):
        print('Iteration', i)
        losses = {}
        # batch up the examples using spaCy's minibatch
        batches = minibatch(train_data, size=compounding(4., 32., 1.001))
        for batch in batches:
            texts, annotations = zip(*batch)
            nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                       losses=losses)

# Evaluate 
train_predictions = np.array([nlp(text).cats['POSITIVE'] > 0.5 for text in train_text])
dev_predictions = np.array([nlp(text).cats['POSITIVE'] > 0.5 for text in dev_text])

print(evaluate(train_labels, train_predictions))
print(evaluate(dev_labels, dev_predictions))

# Evaluate with average weights
print('Evaluation with averaged weights')
with nlp.use_params(optimizer.averages):
    # Evaluate 
    train_predictions = np.array([nlp(text).cats['POSITIVE'] > 0.5 for text in train_text])
    dev_predictions = np.array([nlp(text).cats['POSITIVE'] > 0.5 for text in dev_text])

    print(evaluate(train_labels, train_predictions))
    print(evaluate(dev_labels, dev_predictions))


            




