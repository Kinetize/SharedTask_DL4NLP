import tensorflow as tf


tf.random.set_seed(100)


class Similarity(tf.keras.Model):

    def __init__(self, hidden_units=300, dropout_rate=0.3):
        super(Similarity, self).__init__()

        self.dense_1 = tf.keras.layers.Dense(units=hidden_units, activation='relu')
        self.dense_2 = tf.keras.layers.Dense(units=hidden_units // 2, activation='relu')

        self.dropout_1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.dropout_2 = tf.keras.layers.Dropout(rate=dropout_rate)

        self.out = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, x_, training=False, **kwargs):
        x = tf.reshape(x_, (x_.shape[0], x_.shape[1] * x_.shape[2]))

        x = self.dense_1(x)
        x = self.dropout_1(x, training=training)

        x = self.dense_2(x)
        x = self.dropout_2(x, training=training)

        return self.out(x)
