import numpy as np


def read_dataset(dataset):
    data = ([], [], [])
    with open('data/{}-dataset.txt'.format(dataset), encoding='utf8') as f:
        for l in f:
            splitted = l.strip().split('\t')

            has_score = len(splitted) > 2
            offset = 0 if has_score else 1
            for i in range(len(splitted)):
                data[i + offset].append(float(splitted[i]) if has_score and i == 0 else splitted[i])

    return data


def load_vectors(fname, limit=20000):
    """
    Taken and adapted from https://fasttext.cc/docs/en/english-vectors.html
    """

    fin = open('data/' + fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    n, d = map(int, fin.readline().split())
    data = {}
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.array([float(t) for t in tokens[1:]])

        if i >= limit:
            break

    return data
