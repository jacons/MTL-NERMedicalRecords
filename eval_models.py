from typing import Tuple

import torch

from Configuration import Configuration
from Evaluation.metrics import eval_model
from Parsing.parser_utils import parse_args, holdout, build_dataset
from Training.MTModel import MTBERTClassification

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["bert"])

    if (args.datasets is None) or (args.model is None) or len(args.datasets) != 2:
        raise Exception("Define datasets A and B!")

    paths = args.datasets
    file_model = args.model

    dt, handler_a, handler_b = build_dataset(args.datasets, verbose=True)
    _, _, df_test = holdout(dt)

    model = MTBERTClassification.from_pretrained(conf.bert,
                                                 id2label_a=handler_a.id2label,
                                                 id2label_b=handler_b.id2label)
    model.load_state_dict(torch.load(file_model))

    if conf.cuda:
        model = model.to(conf.gpu)

    result = eval_model(model=model,
                        dataset=df_test,
                        conf=conf,
                        handler_a=handler_a,
                        handler_b=handler_b,
                        result=args.eval)

    if isinstance(result, Tuple):
        print(result[0], result[1])
    else:
        print(result)
