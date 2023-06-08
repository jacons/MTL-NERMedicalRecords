from typing import Optional

from allennlp.modules import ConditionalRandomField
from allennlp.modules.conditional_random_field import allowed_transitions
from torch import Tensor
from torch.nn import Sequential, Dropout, Linear, Module, LeakyReLU, LogSoftmax
from transformers import BertPreTrainedModel, BertModel

"""
secondo me si può snellire il codice facendo facendo bern form pretrained e rimuovendo
uno strato di classe


"""


class ClassifierHead(Module):
    def __init__(self, dropout: float, hidden: int, id2label: dict):
        super(ClassifierHead, self).__init__()

        num_labels = len(id2label)
        self.linear = Sequential(
            LeakyReLU(),
            Dropout(dropout),
            Linear(hidden, num_labels),
            LogSoftmax(-1)
        )

        self.crf_layer = ConditionalRandomField(num_tags=num_labels,
                                                constraints=allowed_transitions(constraint_type="BIO",
                                                                                labels=id2label))

    def forward(self, feature_extracted: Tensor, attention_mask: Tensor, labels: Tensor):
        out = self.linear(feature_extracted[0])

        loss = None
        if labels is not None:
            loss = -self.crf_layer(out, labels, attention_mask) / float(out.size(0))

        best_path = self.crf_layer.viterbi_tags(out, attention_mask)

        output = (best_path,) + feature_extracted[2:]
        return ((loss,) + output) if loss is not None else output


class MTBERTClassification(BertPreTrainedModel):  # noqa

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, id2label_a: dict, id2label_b: dict):
        super().__init__(config)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        # Feature extractor
        self.bert = BertModel(config, add_pooling_layer=False)

        # Multi head
        self.classifierA = ClassifierHead(
            dropout=classifier_dropout,
            hidden=config.hidden_size,
            id2label=id2label_a
        )

        self.classifierB = ClassifierHead(
            dropout=classifier_dropout,
            hidden=config.hidden_size,
            id2label=id2label_b
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
            self,
            # Parameters for token
            input_ids: Optional[Tensor] = None,
            attention_mask: Optional[Tensor] = None,

            # labels
            labels_a: Optional[Tensor] = None,
            labels_b: Optional[Tensor] = None,

            # Other parameters
            token_type_ids: Optional[Tensor] = None,
            position_ids: Optional[Tensor] = None,
            head_mask: Optional[Tensor] = None,
            inputs_embeds: Optional[Tensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
    ):

        feature_extracted = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False,
        )

        out_a = self.classifierA(feature_extracted, attention_mask, labels_a)
        out_b = self.classifierB(feature_extracted, attention_mask, labels_b)

        loss = None
        if labels_a is not None and labels_b is not None:
            loss = out_a[0] + out_b[0]
            best_pathways = (out_a[1], out_b[1])
        else:
            best_pathways = (out_a[0], out_b[0])

        return ((loss,) + best_pathways) if loss is not None else best_pathways


class Classifier(Module):
    def __init__(self, bert: str, id2label_a: dict, id2label_b: dict):
        """
        Bert model
        :param bert: Name of bert used
        """
        super(Classifier, self).__init__()

        self.model = MTBERTClassification.from_pretrained(bert, id2label_a=id2label_a, id2label_b=id2label_b)

        return

    def forward(self, input_id, mask, labels_a, labels_b):
        return self.model(input_ids=input_id, attention_mask=mask, labels_a=labels_a, labels_b=labels_b)
