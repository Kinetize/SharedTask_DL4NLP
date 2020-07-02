from sklearn.metrics import mean_squared_error

from itertools import product
from model import Similarity
from preprocessing import get_datasets
from util import get_similarities, spearman_metric
from adversarial_attacks import VIPER_ICES, VIPER_DCES, VIPER_ECES, DisemvowelingAttack, AttackPipeline


run_best_system = True


def mse_for_ds(dataset):
    similarities = get_similarities(dataset)

    return mean_squared_error(similarities, dataset[1])


if __name__ == "__main__":
    results = []
    configurations = [(True, True, True, False)] if run_best_system else list(product([False, True], repeat=4))

    for config in configurations:
        print(f'Config: {config}')

        attack_pipeline = AttackPipeline([DisemvowelingAttack(p=0.1), VIPER_ECES(p=0.1)])
        train, dev, test, test_scoreboard, test_final = get_datasets([('training', attack_pipeline) if config[3] else 'training',
                                                          ('development', attack_pipeline) if config[3]
                                                          else 'development',
                                                          'test-hex06', 'test-scoreboard', 'test-final'],
                                                         use_edit_distance_correction=config[0],
                                                         use_stop_word_filtering=config[1],
                                                         do_remove_disallowed_chars=config[2],
                                                         use_bert=False)

        train_spearmen_ds = spearman_metric(train[1], get_similarities(train))
        dev_spearman_ds = spearman_metric(dev[1], get_similarities(dev))
        test_spearman_ds = spearman_metric(test[1], get_similarities(test))

        # Train Similarity NN:
        model = Similarity()

        model.compile(optimizer='adam', loss='mse')
        history = model.fit(*train, validation_data=dev, epochs=300, batch_size=100)

        train_mse = model.evaluate(*train, batch_size=100)
        dev_mse = model.evaluate(*dev, batch_size=100)
        test_mse = model.evaluate(*test, batch_size=100)

        train_spearmen = spearman_metric(train[1], [p[0] for p in model.predict(train[0], batch_size=100)])
        dev_spearman = spearman_metric(dev[1], [p[0] for p in model.predict(dev[0], batch_size=100)])
        test_spearman = spearman_metric(test[1], [p[0] for p in model.predict(test[0], batch_size=100)])

        predictions = model.predict(test_scoreboard[0], batch_size=1)
        predictions_final = model.predict(test_final[0], batch_size=1)

        with open('scores_nn.txt', 'w') as f:
            for p in predictions:
                f.write(f'{p[0]:.2f}\n')

        with open('scores_nn_final.txt', 'w') as f:
            for p in predictions_final:
                f.write(f'{p[0]:.2f}\n')

        results.append((0, train_mse, dev_mse, test_mse, 0, train_spearmen, dev_spearman, test_spearman, 0,
                        train_spearmen_ds, dev_spearman_ds, test_spearman_ds))

    with open('results_nn.csv', 'w') as f:
        print('Writing results...')
        for config, res in zip(configurations, results):
            f.write(';'.join(['1' if c else '0' for c in config]) + ';' +
                    ';'.join([f'{s:.4f}' for s in res]) + '\n')
