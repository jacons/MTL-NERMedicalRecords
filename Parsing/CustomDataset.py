from pandas import DataFrame
from torch import LongTensor, IntTensor, BoolTensor
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import BertTokenizerFast

import Configuration
from Parsing.parser_utils import EntityHandler, align_tags


class NerDataset(Dataset):
    # We try to preprocess the data as much as possible before the training.
    def __init__(self, dataset: DataFrame, conf: Configuration,
                 e_handler_a: EntityHandler, e_handler_b: EntityHandler):

        self.list_of_tokens, self.list_of_att_masks, self.list_of_tag_masks = [], [], []

        self.list_of_labels_a, self.list_of_labels_b = [], []

        tokenizer = BertTokenizerFast.from_pretrained(conf.bert)

        for row in tqdm(dataset.itertuples(), total=dataset.shape[0], desc="Building dataset",
                        mininterval=conf.refresh_rate):

            # tokens = ["Hi","How","are","you"]
            tokens, labels_a, labels_b = row[1].split(), row[2].split(), row[3].split()

            token_text = tokenizer(tokens, is_split_into_words=True)

            aligned_labels_a, tag_mask = align_tags(labels_a, token_text.word_ids())
            aligned_labels_b, _ = align_tags(labels_b, token_text.word_ids())

            # prepare a model's inputs
            input_ids = IntTensor(token_text["input_ids"])
            att_mask = IntTensor(token_text["attention_mask"])
            tag_mask = BoolTensor(tag_mask)

            # mapping the list of labels e.g. ["I-DRUG","O"] to list of id of labels e.g. ["4","7"]
            labels_ids_a = LongTensor(e_handler_a.map_lab2id(aligned_labels_a))
            labels_ids_b = LongTensor(e_handler_b.map_lab2id(aligned_labels_b))

            if conf.cuda:
                input_ids = input_ids.to(conf.gpu)
                att_mask = att_mask.to(conf.gpu)
                tag_mask = tag_mask.to(conf.gpu)

                labels_ids_a = labels_ids_a.to(conf.gpu)
                labels_ids_b = labels_ids_b.to(conf.gpu)

            self.list_of_tokens.append(input_ids)
            self.list_of_att_masks.append(att_mask)
            self.list_of_tag_masks.append(tag_mask)

            self.list_of_labels_a.append(labels_ids_a)
            self.list_of_labels_b.append(labels_ids_b)

    def __len__(self):
        return len(self.list_of_tokens)

    def __getitem__(self, idx):

        return (self.list_of_tokens[idx],
                self.list_of_att_masks[idx],
                self.list_of_tag_masks[idx],
                self.list_of_labels_a[idx],
                self.list_of_labels_b[idx])
