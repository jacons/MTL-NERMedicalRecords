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
                 handler_a: EntityHandler, handler_b: EntityHandler):

        self.list_of_tokens, self.list_of_att_masks = [], []

        self.l_tag_masks_a, self.l_labels_a = [], []
        self.l_tag_masks_b, self.l_labels_b = [], []

        tokenizer = BertTokenizerFast.from_pretrained(conf.bert)

        for row in tqdm(dataset.itertuples(), total=dataset.shape[0], desc="Building dataset",
                        mininterval=conf.refresh_rate):

            # tokens = ["Hi","How","are","you"]
            tokens, labels_a, labels_b = row[1].split(), row[2].split(), row[3].split()

            token_text = tokenizer(tokens, is_split_into_words=True)

            align_labels_a, tag_mask_a = align_tags(labels_a, token_text.word_ids())
            align_labels_b, tag_mask_b = align_tags(labels_b, token_text.word_ids())

            # prepare a model's inputs
            input_ids = IntTensor(token_text["input_ids"])
            att_mask = IntTensor(token_text["attention_mask"])

            tag_mask_a = BoolTensor(tag_mask_a)
            tag_mask_b = BoolTensor(tag_mask_b)

            # mapping the list of labels e.g. ["I-DRUG","O"] to list of id of labels e.g. ["4","7"]
            labels_ids_a = LongTensor(handler_a.map_lab2id(align_labels_a))
            labels_ids_b = LongTensor(handler_b.map_lab2id(align_labels_b))

            if conf.cuda:
                input_ids = input_ids.to(conf.gpu)
                att_mask = att_mask.to(conf.gpu)

                tag_mask_a = tag_mask_a.to(conf.gpu)
                tag_mask_b = tag_mask_b.to(conf.gpu)

                labels_ids_a = labels_ids_a.to(conf.gpu)
                labels_ids_b = labels_ids_b.to(conf.gpu)

            self.list_of_tokens.append(input_ids)
            self.list_of_att_masks.append(att_mask)

            self.l_tag_masks_a.append(tag_mask_a)
            self.l_tag_masks_b.append(tag_mask_b)

            self.l_labels_a.append(labels_ids_a)
            self.l_labels_b.append(labels_ids_b)

    def __len__(self):
        return len(self.list_of_tokens)

    def __getitem__(self, idx):

        return (self.list_of_tokens[idx],
                self.list_of_att_masks[idx],

                self.l_tag_masks_a[idx],
                self.l_tag_masks_b[idx],

                self.l_labels_a[idx],
                self.l_labels_b[idx])
