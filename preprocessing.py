from reader import read_dataset, load_vectors
from Levenshtein import distance as levenshtein_distance
import nltk
import numpy as np
from nltk.corpus import stopwords

v = load_vectors('wiki-news-300d-1M.vec', limit=40000)
emb_dim = v['the'].shape[0]

stop_words=set(stopwords.words('english'))


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
    correct_word = ""
    for t in v:
        if levenshtein_distance(token,t) < min_dist:
            min_dist=levenshtein_distance(token,t)
            embedding=v[t]
            correct_word=t
    return correct_word, embedding

def get_embed(token):
    try:
        return token, v[token]
    except KeyError:
        #take the w2v of the most similar word out of the w2v dict -> compute similarity via the edit dist(levenshtein_distance).
        return get_sim_token(token)


def embed_sentence(sentence):
    tokenized = nltk.word_tokenize(sentence)

    cleaned_tokens = []
    embeddings = []
    # get the token embeddings. If the embedding is not in the vec file take the embedding of the most similar word
    for token in tokenized:
        token, embedding = get_embed(token)
        cleaned_tokens.append(token)
        embeddings.append(embedding)

    # now that we have normal words -> delete all the stopwords embeddings
    for index, w in enumerate(cleaned_tokens):
        if w in stop_words:
            embeddings[index] = [0]
    return np.average(np.array(list(filter(lambda x: len(x) > 1, embeddings))), axis=0)
