from typing import Optional, Tuple

from torch import Tensor
from torch.nn import Sequential, Dropout, Linear, CrossEntropyLoss, Module
from transformers import BertPreTrainedModel, BertModel


class MTBERTClassification(BertPreTrainedModel):  # noqa

    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config, num_labels_a: int, num_labels_b: int):
        super().__init__(config)

        self.num_labels_a = num_labels_a  # Number of labels for the task A
        self.num_labels_b = num_labels_b  # Number of labels for the task B

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )

        # Feature extractor
        self.bert = BertModel(config, add_pooling_layer=False)

        # Multi head
        self.classifierA = Sequential(
            Dropout(classifier_dropout),
            Linear(config.hidden_size, num_labels_a)
        )

        self.classifierB = Sequential(
            Dropout(classifier_dropout),
            Linear(config.hidden_size, num_labels_b)
        )

        self.loss_fn = CrossEntropyLoss()
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
    ) -> Tuple[Tensor]:
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

        sequence_output = feature_extracted[0]

        logits_a = self.classifierA(sequence_output)
        logits_b = self.classifierB(sequence_output)

        loss = None
        if labels_a is not None and labels_b is not None:
            # Loss A
            loss_a = self.loss_fn(logits_a.view(-1, self.num_labels_a), labels_a.view(-1))
            # Loss B
            loss_b = self.loss_fn(logits_b.view(-1, self.num_labels_b), labels_b.view(-1))

            loss = 0.5 * (loss_a + loss_b)

        output = (logits_a, logits_b) + feature_extracted[2:]
        return ((loss,) + output) if loss is not None else output


class Classifier(Module):
    def __init__(self, bert: str, num_labels_a: int, num_labels_b: int):
        """
        Bert model
        :param bert: Name of bert used
        """
        super(Classifier, self).__init__()

        self.model = MTBERTClassification.from_pretrained(bert, num_labels_a=num_labels_a, num_labels_b=num_labels_b)

        return

    def forward(self, input_id, mask, labels_a, labels_b):
        output = self.model(input_ids=input_id, attention_mask=mask, labels_a=labels_a, labels_b=labels_b)
        return output
