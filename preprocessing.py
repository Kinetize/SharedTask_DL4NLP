from reader import read_dataset, load_vectors
import Levenshtein
import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk import FreqDist
from nltk.corpus import brown
from gensim.models import KeyedVectors

np.random.seed(100)


use_bert = False
use_weighted_embeddings = True
use_concat_visual_char_embeddings = False
use_edit_distance_correction = True

if not use_bert:
    v = load_vectors('wiki-news-300d-1M.vec', limit=40000)
    emb_dim = v['the'].shape[0]

    embed = lambda sentences: np.array([embed_sentence(sentence) for sentence in sentences])
else:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('bert-base-nli-mean-tokens')
    emb_dim = 768

    embed = lambda sentences: model.encode(sentences)

# Load the visual character embeddings:
# Download here: https://public.ukp.informatik.tu-darmstadt.de/naacl2019-like-humans-visual-attacks/
vce = KeyedVectors.load_word2vec_format("data/vce.normalized")
if use_concat_visual_char_embeddings:
    emb_dim += vce['a'].shape[0]

stop_words=set(stopwords.words('english'))

if use_weighted_embeddings:
    nltk.download('brown')
    fd = FreqDist((word.lower() for word in brown.words()))
    freq_dict = {key: value / fd.N() for (key, value) in dict(fd).items()}

def get_datasets(datasets):
    preprocessed_datasets = []

    for ds in datasets:
        # Load the data set and apply an adversarial attack before if desired:
        if type(ds) is not str:  # Not only dataset path -> adversarial attacker has been appended
            ds, attack = ds
            ds = attack(ds)

        loaded = read_dataset(ds)

        # Remove disallowed special characters (gathered by getting all characters for which str.isalnum yields False):
        disallowed_special_chars = ''.join(['!', '"', '#', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/',
                                            ':', ';', '<', '=', '>', '?', '@', '[', '\\', ']', '^', '_', '`', '{', '|',
                                            '}', '~', '‘', '´', '’', '—', '‚', '–', '“', '”', '˅', '˰', '¶', '˒', '։',
                                            '·', '֊', '˻', '¦', '˯', '˼', '˳', '˗', '”', '˨', '˖', '¡', '˕', '“', '˥',
                                            '˦', '\x12', '˩', '˱', '֍', '¸', '͵', '˿', '˓', '˺', '£', '˭', '˲'])
        visual_equivalents = "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ" + disallowed_special_chars

        def remove_disallowed_chars(text):
            cleaned_text = list()
            replacements = set()
            for sent in text:
                cleaned_sent = list()
                for char in sent:
                    # Replace non-alphanum chars with their alphanum visual equivalent:
                    if not char.isspace() and char not in visual_equivalents:
                        most_sim_alpha_num_char_idx = np.argmax([vce.similarity(char, anc) for anc in visual_equivalents])
                        most_sim_char = visual_equivalents[int(most_sim_alpha_num_char_idx)]
                        replacements.add((char, most_sim_char))
                        char = most_sim_char
                    # Remove disallowed special characters:
                    if char in disallowed_special_chars:
                        char = ''
                    cleaned_sent.append(char)
                cleaned_sent = ''.join(cleaned_sent)
                cleaned_text.append(cleaned_sent)
            print(f"Replaced characters: {', '.join(f'{c} -> {r}' for c, r in sorted(replacements))}")
            return cleaned_text

        # Embed the data:
        embedded = np.array([embed(remove_disallowed_chars(t)) for t in loaded[1:]])
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
    orig_token = token
    try:
        word_embedding = v[token]
    except KeyError:
        try:
            word_embedding = v[token.lower()]
            token = token.lower()
        except KeyError:
            if use_edit_distance_correction:
                #take the w2v of the most similar word out of the w2v dict -> compute similarity via the edit dist(levenshtein_distance).
                token, word_embedding = get_sim_token(token)
                #print(f"Orig: {orig_token}, replaced {token}")
            else:
                word_embedding = np.zeros(emb_dim)

    if use_concat_visual_char_embeddings and len(token) > 0:
        vis_embedding = embed_token_visually(orig_token)  # Using the orig token is crucial to notice orig perturbations
        word_embedding = np.append(vis_embedding, word_embedding)

    return token, word_embedding

def embed_sentence(sentence):
    tokenized = nltk.word_tokenize(sentence)

    cleaned_tokens = []
    embeddings = []
    # get the token embeddings. If the token is not in the vec file take the embedding of the most similar word
    for token in tokenized:
        token, embedding = get_embed(token)
        # sometimes embedding gets an empty list back (only occurs with ratio distances)
        if len(embedding) == 0:
            embeddings.append(np.zeros(emb_dim))
        else:
            embeddings.append(embedding)
        cleaned_tokens.append(token)
    copy_embeddings=embeddings.copy()


    if use_weighted_embeddings:
        # weight the embeddings with their inverse freq_dist
        weights = []
        for w in cleaned_tokens:
            try:
                weights.append(1 / float(freq_dict[w]))
            except:
                weights.append(0.0)
        try:
            return np.average(np.array(embeddings), axis=0, weights=np.array(weights).astype(float))
        except:
            return np.average(np.array(embeddings), axis=0)

    else:
        # now that we have normal words -> delete the stopwords embeddings
        for index, w in enumerate(cleaned_tokens):
            if w in stop_words:
                copy_embeddings[index] = [0]
        filtered = np.array(list(filter(lambda x: len(x) > 1, copy_embeddings)))
        if np.count_nonzero(filtered) > 0:
            return np.average(filtered, axis=0)
        return np.average(embeddings, axis=0)


def embed_token_visually(token):
    return np.mean(vce[list(token)], axis=0)
