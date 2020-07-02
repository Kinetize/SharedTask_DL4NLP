from scipy.stats import spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow as tf


def spearman_metric(y_true, y_pred):
    return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout=tf.float32)


def get_similarities(dataset):
    return np.clip(cosine_similarity(dataset[0][:, 0], dataset[0][:, 1]).diagonal(), 0., 1.)