from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import get_datasets


def get_similarities(dataset):
    return cosine_similarity(dataset[0][:, 0], dataset[0][:, 1]).diagonal()


def mse_for_ds(dataset):
    similarities = get_similarities(dataset)

    return mean_squared_error(similarities, dataset[1])

if __name__ == "__main__":
    train, dev, test, test_scoreboard = get_datasets(['training', 'development', 'test-hex06', 'test-scoreboard'])

    print(f'Train-MSE: {mse_for_ds(train)}')
    print(f'Dev-MSE: {mse_for_ds(dev)}')
    print(f'Test-MSE: {mse_for_ds(test)}')

    similarities = get_similarities(test_scoreboard)

    with open('scores_bert.txt', 'w') as f:
        for p in similarities:
            f.write(f'{p[0]:.2f}\n')
