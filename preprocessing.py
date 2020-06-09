from reader import read_dataset, load_vectors

import nltk
import numpy as np

v = load_vectors('wiki-news-300d-1M.vec', limit=40000)
emb_dim = v['the'].shape[0]


def get_datasets(datasets):
    preprocessed_datasets = []

    for ds in datasets:
        loaded = read_dataset(ds)
        embedded = np.array([[embed_sentence(sentence) for sentence in t] for t in loaded[1:]])
        embedded = np.concatenate([np.expand_dims(embedded[0], axis=1), np.expand_dims(embedded[1], axis=1)], axis=1)
        preprocessed_datasets.append((embedded, np.array(loaded[0])))

    return preprocessed_datasets


def get_embed(token):
    try:
        return v[token]
    except KeyError:
        return np.array([0] * emb_dim)


def embed_sentence(sentence):
    tokenized = nltk.word_tokenize(sentence)

    embeddings = [get_embed(token) for token in tokenized]

    return np.average(embeddings, axis=0)
