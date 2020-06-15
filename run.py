from model import Similarity
from train import train_model
from preprocessing import get_datasets
from adversarial_attacks import VIPER_ICES, VIPER_DCES, VIPER_ECES, DisemvowelingAttack, AttackPipeline

import numpy as np


if __name__ == '__main__':
    run_eagerly = False

    train, dev, test, test_scoreboard = get_datasets([('training', AttackPipeline([DisemvowelingAttack(p=0.2),
                                                                                   VIPER_ICES(p=0.1)])),
                                                      'development', 'test-hex06', 'test-scoreboard'])

    model = Similarity()

    if run_eagerly:
        train_model(model, train[0], np.array(train[0]))
    else:
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        history = model.fit(*train, validation_data=dev, epochs=300, batch_size=100)

        import matplotlib.pyplot as plt
        training_loss = history.history['loss']
        dev_loss = history.history['val_loss']
        plt.plot(training_loss, 'r--')
        plt.plot(dev_loss, 'b-')
        plt.legend(['Training Loss', 'Dev Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()

    print(model.evaluate(*train, batch_size=100))
    print(model.evaluate(*dev, batch_size=100))
    print(model.evaluate(*test, batch_size=100))
    predictions = model.predict(test_scoreboard[0], batch_size=100)

    with open('out.txt', 'w') as f:
        for p in predictions:
            f.write(f'{p[0]}\n')
