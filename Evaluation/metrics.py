import sys
from io import StringIO

import torch
from pandas import DataFrame
from torch import Tensor, zeros, IntTensor, BoolTensor, LongTensor, masked_select, no_grad
from torch.nn import Module
from tqdm import tqdm
from transformers import BertTokenizerFast

import Configuration
from Evaluation.conlleval import evaluate
from Parsing.parser_utils import EntityHandler, align_tags


def scores(confusion: Tensor, all_metrics=False):
    """
    Given a Confusion matrix, returns an F1-score if all_metrics is false, then returns only a mean of F1-score
    """
    length = confusion.shape[0]
    iter_label = range(length)

    accuracy = zeros(length)
    precision = zeros(length)
    recall = zeros(length)
    f1 = zeros(length)

    for i in iter_label:
        fn = torch.sum(confusion[i, :i]) + torch.sum(confusion[i, i + 1:])  # false negative
        fp = torch.sum(confusion[:i, i]) + torch.sum(confusion[i + 1:, i])  # false positive
        tn, tp = 0, confusion[i, i]  # true negative, true positive

        for x in iter_label:
            for y in iter_label:
                if (x != i) & (y != i):
                    tn += confusion[x, y]

        accuracy[i] = (tp + tn) / (tp + fn + fp + tn)
        precision[i] = tp / (tp + fp)
        recall[i] = tp / (tp + fn)
        f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    if all_metrics:
        metrics_dict = {
            "Accuracy": accuracy.tolist(),
            "Precision": precision.tolist(),
            "Recall": recall.tolist(),
            "F1": f1.tolist()
        }
        return DataFrame(metrics_dict)
    else:
        return f1.mean()


def eval_model(model: Module, dataset: DataFrame, conf: Configuration, handler_a: EntityHandler,
               handler_b: EntityHandler, result="conlleval"):
    model.eval()
    confusion_a = zeros(size=(len(handler_a), len(handler_a)))
    confusion_b = zeros(size=(len(handler_b), len(handler_b)))
    true_label_a, pred_label_a = [], []  # using for conlleval
    true_label_b, pred_label_b = [], []  # using for conlleval

    tokenizer = BertTokenizerFast.from_pretrained(conf.bert)
    with no_grad():
        for row in tqdm(dataset.itertuples(), total=dataset.shape[0], desc="Evaluating", mininterval=conf.refresh_rate):

            # ========== Preprocessing ==========
            # tokens = ["Hi","How","are","you"], labels = ["O","I-TREAT" ...]
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
                input_ids = input_ids.to(conf.gpu).unsqueeze(0)
                att_mask = att_mask.to(conf.gpu).unsqueeze(0)

                tag_mask_a = tag_mask_a.to(conf.gpu)
                tag_mask_b = tag_mask_b.to(conf.gpu)

                labels_ids_a = labels_ids_a.to(conf.gpu)
                labels_ids_b = labels_ids_b.to(conf.gpu)
            # ========== Preprocessing ==========

            # ========== Evaluating ==========
            # Perform the prediction
            path_a, path_b = model(input_ids, att_mask)

            if conf.cuda:
                path_a = path_a.to(conf.gpu)
                path_b = path_b.to(conf.gpu)

            labels_a = masked_select(labels_ids_a, tag_mask_a)
            path_a = masked_select(path_a, tag_mask_a)

            for lbl, pre in zip(labels_a, path_a):
                confusion_a[lbl, pre] += 1

            labels_b = masked_select(labels_ids_b, tag_mask_b)
            path_b = masked_select(path_b, tag_mask_b)

            for lbl, pre in zip(labels_b, path_b):
                confusion_b[lbl, pre] += 1
            # ========== Evaluating ==========

            labels_a = handler_a.map_id2lab(labels_a, is_tensor=True)
            labels_b = handler_b.map_id2lab(labels_b, is_tensor=True)

            path_a = handler_a.map_id2lab(path_a, is_tensor=True)
            path_b = handler_b.map_id2lab(path_b, is_tensor=True)

            true_label_a.extend(labels_a)
            true_label_b.extend(labels_b)

            pred_label_a.extend(path_a)
            pred_label_b.extend(path_b)

    if result == "conlleval":

        old_stdout = sys.stdout
        sys.stdout = results = StringIO()

        # ConLL script evaluation https://github.com/sighsmile/conlleval
        evaluate(true_label_a, pred_label_a)
        evaluate(true_label_b, pred_label_b)
        sys.stdout = old_stdout

        return results.getvalue()

    else:
        result_a = scores(confusion_a, all_metrics=True)
        result_b = scores(confusion_b, all_metrics=True)

        result_a.index = handler_a.map_id2lab([*range(0, len(handler_a))])
        result_b.index = handler_b.map_id2lab([*range(0, len(handler_b))])

        return result_a, result_b
