import argparse
from itertools import groupby
from typing import Tuple

import numpy as np
from pandas import DataFrame


def read_conll(path: str):
    """
    Generator of sentences from CoNLL files
    :param path: a path of file
    :return: (sentence string, sequence of label)
    """

    def _is_divider(line: str) -> bool:
        return True if line.strip() == '' else False

    with open(path, encoding="utf-8") as f:
        for is_divider, lines in groupby(f, _is_divider):
            if is_divider:
                continue
            fields = [line.split() for line in lines if not line.startswith('-DOCSTART-')]
            if len(fields) == 0:
                continue
            tokens, entities = [], []
            for row in fields:
                # Sometimes there are more "words" associated to a single tag. Es "date"
                half = int(len(row) / 2) - 1
                token, entity = row[:half], row[-1]
                tokens.append("-".join(token))
                entities.append(entity)
            yield tokens, entities


class EntityHandler:
    """
    EntityHandler is class used to keep the associations between labels and ids
    """

    def __init__(self, set_entities: set):
        self.set_entities = set_entities  # set of all entity detected

        # Give a label returns id : label --> id
        self.label2id: dict = {k: v for v, k in enumerate(sorted(set_entities))}
        # Give id returns a label : id --> label
        self.id2label: dict = {v: k for v, k in enumerate(sorted(set_entities))}

    def map_lab2id(self, list_of_labels, is_tensor=False) -> list:
        """
        Mapping a list of labels into a list of label's id
        """
        result = []
        for label in list_of_labels:
            label = label.item() if is_tensor else label
            result.append(self.label2id[label] if label in self.label2id else self.label2id["O"])

        return result

    def map_id2lab(self, list_of_ids, is_tensor=False) -> list:
        """
        Mapping a list of ids into a list of labels
        """
        result = []
        for label_id in list_of_ids:
            label_id = label_id.item() if is_tensor else label_id
            result.append(self.id2label[label_id] if label_id in self.id2label else "O")

        return result


def build_dataset(path_files: tuple[str, str], verbose=True) -> Tuple[DataFrame, EntityHandler, EntityHandler]:
    """
        Build the dataframe of sentences from a path of file
        The dataframe is composed by two columns "sentences" and "labels"
    """
    sentences = []
    list_of_labels_a = []
    list_of_labels_b = []
    set_entities_a = set()  # set of unique entity found (incrementally updated)
    set_entities_b = set()  # set of unique entity found (incrementally updated)

    gen_a = read_conll(path_files[0])  # Generator for dataset A
    gen_b = read_conll(path_files[1])  # Generator for dataset B

    for fields_a, fields_b in zip(gen_a, gen_b):  # generator

        tokens_a, labels_a = fields_a[0], fields_a[1]
        labels_b = fields_b[1]

        sentences.append(" ".join(tokens_a))  # Concat tokens

        list_of_labels_a.append(" ".join(labels_a))  # Concat label of dataset A
        list_of_labels_b.append(" ".join(labels_b))  # Concat label of dataset B

        set_entities_a.update(labels_a)  # to keep track the entity found
        set_entities_b.update(labels_b)  # to keep track the entity found

    if verbose:
        print("Building sentences and tags\n" + "-" * 85)
        print(f"|{'Sentences':^41}|{'Tags':^41}|")
        print(f"|{'':-^41}|{'':-^41}|")
        print(f"|{len(sentences):^41}|{len(set_entities_a):^20}|{len(set_entities_b):^20}|")
        print(f"|{'':-^41}|{'':-^41}|")
        print(f"|{' - '.join(sorted(set_entities_a)):^83}|")
        print(f"|{' - '.join(sorted(set_entities_b)):^83}|")
        print(f"{'-' * 85}")

    df_data = {"sentences": sentences, "labels_a": list_of_labels_a, "labels_b": list_of_labels_b}
    return DataFrame(df_data).drop_duplicates(), EntityHandler(set_entities_a), EntityHandler(set_entities_b)


def holdout(df: DataFrame, size: float = 1, verbose=True) -> DataFrame:
    """
    Dividing the dataset based on holdout technique
    """
    # ========== PARAMETERS ==========
    tr_size: float = 0.8  # Training set dimensions
    vl_size: float = 0.1  # Validation set dimensions
    ts_size: float = 0.1  # Test set dimensions
    # ========== PARAMETERS ==========

    # Apply a subsampling to reduce the dimension of dataset, it also shuffles the dataset
    # we fixed the random state for the determinism
    df = df.sample(frac=size, random_state=71)

    length = len(df)  # length of sub-sampled dataframe
    tr = int(tr_size * length)  # Number of rows for the training set
    vl = int(vl_size * length)  # validation
    ts = int(ts_size * length)  # test

    if verbose:
        print("|{:^27}|{:^27}|{:^27}|".format("TR: " + str(tr), "VL: " + str(vl),
                                              "TS: " + str(ts)) + "\n" + "-" * 85)
    return np.split(df, [tr, int((tr_size + vl_size) * length)])


def align_tags(labels: list, word_ids: list):
    """
    This function aligns the labels associated with a sentence and returns an "aligned" list and a "tag mask".

    "Aligned" list: represents a list of labels that are aligned with the word-pieces of the sentence,
                    if a token is split in more than one sub-word, the tag associated is repeated.
                    The second tag (the sequence of sub-words) always starts with "I-".

    "Tag mask": represents a list of booleans, where each value corresponding to a sub-token. The boolean
                    is true if the sub-token is the first one (of sub-words) else false.

    *WordPiece is a sub-word segmentation algorithm
    """
    aligned_labels = []
    mask = [False] * len(word_ids)
    prev_id = None

    for idx, word_id in enumerate(word_ids):

        if word_id is None:
            aligned_labels.append("O")

        elif word_id != prev_id:
            aligned_labels.append(labels[word_id])
            mask[idx] = True

        elif word_id == prev_id:
            if labels[word_id][0] == "B":
                aligned_labels.append("I" + labels[word_id][1:])
            else:
                aligned_labels.append(labels[word_id])

        prev_id = word_id
    return aligned_labels, mask


def parse_args():
    p = argparse.ArgumentParser(description='Model configuration.', add_help=True)

    p.add_argument('--datasets', type=str, nargs='+',
                   help='Dataset used for training, it will split in training, validation and test', default=None)

    p.add_argument('--model', type=str,
                   help='Model trained ready to evaluate or use', default=None)

    p.add_argument('--model_name', type=str,
                   help='Name to give to a trained model', default=None)

    p.add_argument('--path_model', type=str,
                   help='Directory to save the model', default=".")

    p.add_argument('--bert', type=str,
                   help='Bert model provided by Huggingface', default="dbmdz/bert-base-italian-xxl-cased")

    p.add_argument('--save', type=int,
                   help='set 1 if you want save the model otherwise set 0', default=1)

    p.add_argument('--eval', type=str,
                   help='define the type of evaluation: conlleval or df', default="conlleval")

    p.add_argument('--lr', type=float, help='Learning rate', default=0.001)

    p.add_argument('--momentum', type=float, help='Momentum', default=0.9)

    p.add_argument('--weight_decay', type=float, help='Weight decay', default=0)

    p.add_argument('--batch_size', type=int, help='Batch size', default=2)

    p.add_argument('--max_epoch', type=int, help='Max number of epochs', default=20)

    p.add_argument('--patience', type=float, help='Patience in early stopping', default=3)

    p.add_argument('--refresh_rate', type=int, help='Refresh rate in tqdm', default=60)

    return p.parse_known_args()
