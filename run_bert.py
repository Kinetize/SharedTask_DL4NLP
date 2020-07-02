from itertools import product
from sklearn.metrics import mean_squared_error

from preprocessing import get_datasets
from util import get_similarities, spearman_metric
from adversarial_attacks import VIPER_ICES, VIPER_DCES, VIPER_ECES, DisemvowelingAttack, AttackPipeline


run_adverserial = False
run_best_system = True


def mse_for_ds(dataset):
    similarities = get_similarities(dataset)

    return mean_squared_error(similarities, dataset[1])


if __name__ == "__main__":
    results = []
    configurations = [(True, False, True)] if run_best_system else list(product([False, True], repeat=3))

    for config in configurations:
        print(f'Config: {config}')

        attack_pipeline = AttackPipeline([DisemvowelingAttack(p=0.1), VIPER_ECES(p=0.1)])
        dev, test, test_scoreboard, test_final = get_datasets([('development', attack_pipeline)
                                                               if run_adverserial else 'development', 'test-hex06',
                                                               'test-scoreboard', 'test-final'],
                                                  use_edit_distance_correction=config[0],
                                                  use_stop_word_filtering=config[1],
                                                  do_remove_disallowed_chars=config[2],
                                                  use_bert=True)

        dev_mse = mse_for_ds(dev)
        test_mse = mse_for_ds(test)

        dev_spearman = spearman_metric(dev[1], get_similarities(dev))
        test_spearman = spearman_metric(test[1], get_similarities(test))

        similarities = get_similarities(test_scoreboard)
        similarities_final = get_similarities(test_final)

        with open('scores_bert.txt', 'w') as f:
            for p in similarities:
                f.write(f'{p:.2f}\n')

        with open('scores_bert_final.txt', 'w') as f:
            for p in similarities_final:
                f.write(f'{p:.2f}\n')

        results.append((0, 0, dev_mse, test_mse, 0, 0, dev_spearman, test_spearman))

    with open('results_bert.csv', 'w') as f:
        print('Writing results...')
        for config, res in zip(configurations, results):
            f.write(';'.join(['1' if c else '0' for c in config]) + ';' +
                    ';'.join([f'{s:.4f}' for s in res]) + '\n')
