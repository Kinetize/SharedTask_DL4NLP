from model import Similarity
from train import train_model
from preprocessing import get_datasets
from adversarial_attacks import VIPER_ICES, VIPER_DCES, VIPER_ECES, DisemvowelingAttack, AttackPipeline
from scipy.stats import spearmanr
import tensorflow as tf
import numpy as np

np.random.seed(100)


def spearman_metric(y_true, y_pred):
    return tf.py_function(spearmanr, [tf.cast(y_pred, tf.float32), tf.cast(y_true, tf.float32)], Tout=tf.float32)


if __name__ == '__main__':
    run_eagerly = False

    attack_pipeline = AttackPipeline([DisemvowelingAttack(p=0.1), VIPER_ECES(p=0.1)])
    train, dev, test, test_scoreboard = get_datasets(['training',
                                                      #('training', attack_pipeline),
                                                      'development',
                                                      #('development', attack_pipeline),
                                                      'test-hex06', 'test-scoreboard'])

    model = Similarity()

    if run_eagerly:
        train_model(model, train[0], np.array(train[0]))
    else:
        model.compile(optimizer='adam', loss='mse', metrics=['mae', spearman_metric])
        history = model.fit(*train, validation_data=dev, epochs=400, batch_size=100)

        import matplotlib.pyplot as plt
        # Plot train & dev evaluation curve on different metrics:
        for metric in ['loss', 'spearman_metric']:
            training_loss = history.history[metric]
            dev_loss = history.history[f"val_{metric}"]
            plt.plot(training_loss, 'r--')
            plt.plot(dev_loss, 'b-')
            plt.legend(['Training set', 'Dev set'])
            plt.xlabel('Epoch')
            plt.ylabel(metric)
            plt.title(metric)
            plt.show()

    print(model.evaluate(*train, batch_size=100))
    print(model.evaluate(*dev, batch_size=100))
    print(model.evaluate(*test, batch_size=100))
    predictions = model.predict(test_scoreboard[0], batch_size=100)

    with open('scores.txt', 'w') as f:
        for p in predictions:
            f.write(f'{p[0]:.2f}\n')
