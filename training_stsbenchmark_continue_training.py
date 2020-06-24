"""
This example loads the pre-trained SentenceTransformer model 'bert-base-nli-mean-tokens' from the server.
It then fine-tunes this model for some epochs on the STS benchmark dataset.

Note: In this example, you must specify a SentenceTransformer model.
If you want to fine-tune a huggingface/transformers model like bert-base-uncased, see training_nli.py and training_stsbenchmark.py
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import STSBenchmarkDataReader
import logging
from datetime import datetime
from reader import read_dataset
from sentence_transformers.readers import InputExample
from adversarial_attacks import VIPER_ICES, VIPER_DCES, VIPER_ECES, DisemvowelingAttack, AttackPipeline


def get_datasets(datasets):
    preprocessed_datasets = []

    for d_idx, ds in enumerate(datasets):
        # Load the data set and apply an adversarial attack before if desired:
        if type(ds) is not str:  # Not only dataset path -> adversarial attacker has been appended
            ds, attack = ds
            ds = attack(ds)

        loaded = read_dataset(ds)
        converted = list()
        if len(loaded[0]) == 0:  # Unupervised -> pad labels with -1
            loaded = ([-1] * len(loaded[1]), *loaded[1:])
        for line_idx, line in enumerate(zip(*loaded)):
            input_example = InputExample(guid=f"{d_idx}_{line_idx}", texts=list(line[-2:]), label=float(line[0]))
            converted.append(input_example)

        preprocessed_datasets.append(converted)

    return preprocessed_datasets


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
model_name = 'data/roberta-base-nli-stsb-mean-tokens'
train_batch_size = 16
num_epochs = 4
model_save_path = 'data/training_stsbenchmark_continue_training_AA-'+model_name+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

attack_pipeline = AttackPipeline([DisemvowelingAttack(p=0.1),
                                  VIPER_ECES(p=0.1)
                                  ])
train, dev, test, test_scoreboard = get_datasets([#'training',
                                                      ('training', attack_pipeline),
                                                      #'development',
                                                      ('development', attack_pipeline),
                                                      'test-hex06', 'test-scoreboard'])

#sts_reader = STSBenchmarkDataReader('../datasets/stsbenchmark', normalize_scores=True)

# Load a pre-trained sentence transformer model
model = SentenceTransformer(model_name)

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(train, model)#SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(dev, model)#SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SentenceTransformer(model_save_path)
test_data = SentencesDataset(test, model)# SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
model.evaluate(evaluator)
