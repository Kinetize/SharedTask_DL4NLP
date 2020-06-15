import os
import subprocess
from abc import abstractmethod
from queue import SimpleQueue

from adversarial_attacks.basics import AdversarialAttackBase


class VIPERAttackBase(AdversarialAttackBase):
    def __init__(self, p):
        super(VIPERAttackBase, self).__init__()

        # All VIPER attacks have a parameter p that determines the probability of a character being manipulated:
        assert 0.0 <= p <= 1.0, f"p = {p} is no valid probability value between 0 and 1!"
        self.p = p

        # misc
        self._char_sep_len = None
        self._sim_scores = None

    @property
    def attack_code(self):
        return f"{self.__class__.__name__}(p={self.p})"  # As p is a important parameter, append it to the attack code

    @abstractmethod
    def invoke_viper_sub_process(self):
        ...  # Needs to be implemented in subclasses!

    def _attack(self, dataset_path, perturbed_dataset_path):
        sub_proc = self.invoke_viper_sub_process()
        self.dataset_file_stream(sub_proc,
                                 in_file_path=dataset_path,
                                 out_file_path=perturbed_dataset_path)

    def dataset_file_stream(self, sub_proc, in_file_path, out_file_path):
        self._sim_scores = SimpleQueue()  # Init queue

        in_str = self._in_stream(in_file_path)
        self._out_stream(sub_proc.communicate(in_str)[0], out_file_path)

    def _in_stream(self, in_file_path):
        buffer = list()

        with open(in_file_path, 'r', encoding='utf-8') as in_file_handle:
            for line in in_file_handle:
                line_split = line.split("\t")

                # Store similarity score in queue if data set is labeled:
                if len(line_split) == 3:
                    self._sim_scores.put(float(line_split[0]))

                # Append sentences to buffer:
                sentence_a, sentence_b = line_split[-2:]
                buffer.append(sentence_a.strip())
                buffer.append(sentence_b.strip())

        return "\n".join(buffer)

    def _out_stream(self, stream, out_file_path):
        with open(out_file_path, 'w', encoding='utf-8') as out_file_handle:
            sentence_a, sentence_b = None, None

            for line in stream.split("\n"):
                if len(line) == 0:  # Skip empty lines
                    continue
                line = line.split("\t")[0]  # Remove any tab-separated additional sentences (if present)

                if self._char_sep_len is not None:  # Removes character separations if needed
                    line = line[::self._char_sep_len + 1]

                # The lines are alternately assigned to record A or record B:
                if sentence_a is None:
                    sentence_a = line
                else:
                    sentence_b = line

                    # Restore similarity score from queue if data set is labeled:
                    if not self._sim_scores.empty():
                        score = self._sim_scores.get()
                        line = f"{score}\t{sentence_a}\t{sentence_b}\n"
                    else:
                        line = f"{sentence_a}\t{sentence_b}\n"
                    out_file_handle.write(line)
                    sentence_a, sentence_b = None, None  # Clean sentences for next iteration


class VIPER_ICES(VIPERAttackBase):

    def __init__(self, p,
                 embeddings_file_path="data/vce.normalized", perturbations_file="data/perturbations_file_ices.txt"):
        super(VIPER_ICES, self).__init__(p=p)

        self.embeddings_file_path = embeddings_file_path
        self.perturbations_file = perturbations_file

    def invoke_viper_sub_process(self):
        # Download visual character embeddings file (vce.normalized) from
        # https://public.ukp.informatik.tu-darmstadt.de/naacl2019-like-humans-visual-attacks/
        sub_proc = subprocess.Popen(["python", "UKP_Visual_Attacks/code/VIPER/viper_ices.py",
                                     "-e", self.embeddings_file_path,
                                     "--perturbations-file", self.perturbations_file,
                                     "-p", f"{self.p}"],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        return sub_proc


class VIPER_DCES(VIPERAttackBase):

    def __init__(self, p, perturbations_file="data/perturbations_file_dces.txt"):
        super(VIPER_DCES, self).__init__(p=p)

        self.perturbations_file = perturbations_file

        self._tmp_in_file_path = "data/TMP_DCES.txt"  # VIPER_DCES requires the input file path as argument
        self._char_sep_len = 1  # Counteract VIPER_DCES inserting a bank separator (of length 1) between each character

    def _in_stream(self, in_file_path):
        in_str = super(VIPER_DCES, self)._in_stream(in_file_path=in_file_path)

        # VIPER_DCES requires temporary input file (instead of input stream)  ->  Write this file
        with open(self._tmp_in_file_path, 'w', encoding='utf-8') as tmp_in_file_handle:
            tmp_in_file_handle.write(in_str)
        in_str = None

        return in_str

    def _out_stream(self, stream, out_file_path):
        super(VIPER_DCES, self)._out_stream(stream=stream, out_file_path=out_file_path)

        # VIPER_DCES requires temporary input file (instead of input stream)  ->  Clean up now
        if self._tmp_in_file_path is not None:
            os.remove(self._tmp_in_file_path)

    def invoke_viper_sub_process(self):
        sub_proc = subprocess.Popen(["python", "UKP_Visual_Attacks/code/VIPER/viper_dces.py",
                                     "-d", self._tmp_in_file_path,
                                     "--perturbations-file", self.perturbations_file,
                                     "-p", f"{self.p}"],
                                    stdout=subprocess.PIPE, text=True, encoding='utf-8')
        return sub_proc


class VIPER_ECES(VIPERAttackBase):

    def __init__(self, p, selected_neighbors_file="UKP_Visual_Attacks/code/VIPER/selected.neighbors"):
        super(VIPER_ECES, self).__init__(p=p)

        self.selected_neighbors_file = selected_neighbors_file

    def invoke_viper_sub_process(self):
        sub_proc = subprocess.Popen(["python", "UKP_Visual_Attacks/code/VIPER/viper_eces.py",
                                     f"{self.p}", self.selected_neighbors_file],
                                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, encoding='utf-8')
        return sub_proc
