from reader import read_dataset, load_vectors
import Levenshtein
import nltk
import numpy as np
from nltk.corpus import stopwords

np.random.seed(100)


use_bert = False

if not use_bert:
    v = load_vectors('wiki-news-300d-1M.vec', limit=40000)
    emb_dim = v['the'].shape[0]

    embed = lambda sentences: np.array([embed_sentence(sentence) for sentence in sentences])
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    emb_dim = 768

    embed = lambda sentences: model.encode(sentences)

stop_words=set(stopwords.words('english'))


def get_datasets(datasets):
    preprocessed_datasets = []

    for ds in datasets:
        # Load the data set and apply an adversarial attack before if desired:
        if type(ds) is not str:  # Not only dataset path -> adversarial attacker has been appended
            ds, attack = ds
            ds = attack(ds)

        loaded = read_dataset(ds)

        # Remove disallowed special characters:
        disallowed_special_chars = ['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':',
                                    ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|', '}', '~']

        def remove_disallowed_special_chars(text):
            return [''.join(char for char in sent if char not in disallowed_special_chars) for sent in text]

        # Embed the data:
        embedded = np.array([embed(remove_disallowed_special_chars(t)) for t in loaded[1:]])
        embedded = np.concatenate([np.expand_dims(embedded[0], axis=1), np.expand_dims(embedded[1], axis=1)], axis=1)
        preprocessed_datasets.append((embedded, np.array(loaded[0])))

    return preprocessed_datasets


def get_sim_token(token):
    embedding=[]
    max_ratio=0
    correct_word = ""
    for t in v:
        ratio=Levenshtein.jaro(token,t)
        if ratio > max_ratio:
            max_ratio=ratio
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
    # get the token embeddings. If the token is not in the vec file take the embedding of the most similar word
    for token in tokenized:
        token, embedding = get_embed(token)
        cleaned_tokens.append(token)
        embeddings.append(embedding)
    copy_embeddings=embeddings.copy()
    # now that we have normal words -> delete the stopwords embeddings
    for index, w in enumerate(cleaned_tokens):
        if w in stop_words:
            copy_embeddings[index] = [0]
    filtered = np.array(list(filter(lambda x: len(x) > 1, copy_embeddings)))
    if np.count_nonzero(filtered) > 0:
        return np.average(filtered, axis=0)
    return np.average(embeddings, axis=0)

