from adversarial_attacks.basics import AdversarialAttackBase


class AttackPipeline(AdversarialAttackBase):
    """
    Conveniently pipelines multiple attacks (automatically cleans intermediate data files, etc).
    """

    def __init__(self, attacks):
        super(AttackPipeline, self).__init__()

        self.attacks = attacks

    def attack(self, dataset, perturbed_dataset=None):
        assert perturbed_dataset is None, "AttackPipeline does not support custom naming of the perturbed dataset."

        # Iteratively perform attacks in order to build the final perturbed dataset
        perturbed_dataset = dataset
        for attack in self.attacks:
            perturbed_dataset = attack(perturbed_dataset)
        assert self.attack_code in perturbed_dataset, self.attack_code  # Sanity check

        return perturbed_dataset

    def _attack(self, dataset_path, perturbed_dataset_path):
        pass  # Nothing needs to be done here since AttackPipeline is no real attack but a wrapper class -> self.attack

    @property
    def attack_code(self):
        return "-".join(attack.attack_code for attack in self.attacks)
