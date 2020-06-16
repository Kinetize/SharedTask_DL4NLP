import numpy as np
import tensorflow as tf
np.random.seed(100)
tf.random.set_seed(100)


def train_model(model, train_in, train_labels, batch_size=100, epochs=300):
    # Instantiate an optimizer.
    optimizer = tf.keras.optimizers.Adam()
    loss_fn = tf.keras.losses.MeanSquaredError()
    loss_metric = tf.keras.metrics.Mean()

    # Iterate over the batches of a dataset.
    for epoch in range(epochs):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step in range(train_in.shape[0] // batch_size):
            x_batch_train = train_in[batch_size * step:batch_size * (step + 1)]
            y_batch_train = train_labels[batch_size * step:batch_size * (step + 1)]

            with tf.GradientTape() as tape:
                reconstructed = model(x_batch_train, training=True)

                loss = 0
                for i, t, r in zip(range(len(y_batch_train)), y_batch_train, reconstructed):
                    loss += loss_fn(t, r)

                loss /= x_batch_train.shape[1]
                loss += np.sum(model.losses)

            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            loss_metric(loss)

            if step % 10 == 0:
                print('step %s: mean loss = %s' % (step, loss_metric.result()))
