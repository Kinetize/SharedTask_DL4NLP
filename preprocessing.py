from reader import read_dataset, load_vectors
from Levenshtein import distance as levenshtein_distance
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


def get_sim_token(token):
    embedding=[]
    min_dist=999
    for t in v:
        if levenshtein_distance(token,t) < min_dist:
            min_dist=levenshtein_distance(token,t)
            embedding=v[t]
    return embedding

def get_embed(token):
    try:
        return v[token]
    except KeyError:
        #take the w2v of the most similar word out of the w2v dict -> compute similarity via the edit dist(levenshtein_distance).
        return get_sim_token(token)


def embed_sentence(sentence):
    tokenized = nltk.word_tokenize(sentence)

    embeddings = [get_embed(token) for token in tokenized]

    return np.average(embeddings, axis=0)
