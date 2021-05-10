import re

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer


def tokenize(x):
    '''Tokenize a list of sentences `x` into a list of lists where each sublist
    contains the tokens from each starting sentence'''

    x = [word_tokenize(s) for s in x]

    return x

def tokenized_to_lower(x):
    '''Expects a list of lists `x` containing tokenized words. The function will return another 
    list of lists the same size as the input, with all tokens in lower case'''

    x = [[str(w).lower() for w in s] for s in x]

    return x

def remove_symbols(x, pattern = r'[^a-zA-Z0-9]', replacewith = '',):
    '''Removes or replaces all symbols from strings in `x` given by the regex pattern `pattern`.
      `x` is expected to be a list of lists'''

    x = [[re.sub(pattern, replacewith, w) for w in s] for s in x]

    return x

def remove_non_alpha(x):
    '''Expects a list of lists `x` with each sublist containing tokenized
    strings. Removes any string containing non-alphabet characters.'''

    x = [[w for w in s if w.isalpha()] for s in x]

    return x

def remove_stopwords(x):
    '''Remove stopwords from the list of lists `x`. Uses the nltk English
    language stopwords corpus'''
    
    stop = set(stopwords.words('english'))
    x = [[w for w in s if w not in stop] for s in x]

    return x
 
def token_stemming(x):
    '''Uses the LancasterStemmer from nltk to stem tokens in the provided
    list of lists `x`'''

    stemmer = LancasterStemmer()
    x = [[stemmer.stem(w) for w in s] for s in x]
    
    return x

def process_texts(x):
    x = tokenize(x)
    x = tokenized_to_lower(x)
    x = remove_symbols(x)
    x = remove_non_alpha(x)
    x = remove_stopwords(x)
    x = token_stemming(x)
    return x