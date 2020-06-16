import os
import random
from abc import ABC, abstractmethod


class AdversarialAttackBase(ABC):
    """
    (Abstract) Base attack class
    """
    def attack(self, dataset, perturbed_dataset=None):

        # Automatically infers the perturbed dataset name if not specified:
        if perturbed_dataset is None:
            perturbed_dataset = f"{dataset}-{self.attack_code}"
        perturbed_dataset_path = f"data/{perturbed_dataset}-dataset.txt"

        if os.path.isfile(perturbed_dataset_path):  # Skip attack if perturbed dataset file does already exist
            print(f"Adversarial attack {self.attack_code} on {dataset}: "
                  f"Pre-computed perturbed data ({perturbed_dataset}) is reused.")
        else:
            # Apply the adversarial attack:
            self._attack(dataset_path=f"data/{dataset}-dataset.txt",
                         perturbed_dataset_path=perturbed_dataset_path)

        return perturbed_dataset

    @abstractmethod
    def _attack(self, dataset_path, perturbed_dataset_path):
        ...  # Needs to be implemented in subclasses!

    def __call__(self, dataset, perturbed_dataset=None):
        return self.attack(dataset=dataset, perturbed_dataset=perturbed_dataset)

    @property
    def attack_code(self):
        return self.__class__.__name__


class DisemvowelingAttack(AdversarialAttackBase):

    vowels = "aeiouAEIOU"

    def __init__(self, p):
        super(DisemvowelingAttack, self).__init__()

        assert 0.0 <= p <= 1.0, f"p = {p} is no valid probability value between 0 and 1!"
        self.p = p

    def _attack(self, dataset_path, perturbed_dataset_path):
        with open(dataset_path, 'r', encoding='utf-8') as in_handle, \
             open(perturbed_dataset_path, 'w', encoding='utf-8') as out_handle:
            for line in in_handle:
                perturbed_line = ""
                line_split = line.split('\t')

                if len(line_split) == 3:  # Supervised data set
                    perturbed_line = f"{line_split[0]}\t"  # Put the label at the beginning of the perturbed line

                # Perturb sentences:
                sentences = '\t'.join(line_split[-2:])
                perturbed_line += self._disemvowel(sentences)

                out_handle.write(perturbed_line)

    def _disemvowel(self, text):
        # Keep character if it is not a vowel or random number (between 0 and 1) is greater than p
        # (which is equivalent to remove a character with probability p if it is a vowel)
        return ''.join(char for char in list(text)
                       if char not in self.vowels or random.random() > self.p)

    @property
    def attack_code(self):
        return f"{self.__class__.__name__}(p={self.p})"  # As p is a important parameter, append it to the attack code
