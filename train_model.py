from Configuration import Configuration
from Parsing.parser_utils import parse_args, build_dataset, holdout
from Training.MTModel import MTBERTClassification
from Training.Trainer import train

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["param", "bert"])

    if args.model_name is None:
        raise Exception("Define a model name!")

    dt, handlerA, handlerB = build_dataset(args.datasets, verbose=True)

    df_train, df_val, _ = holdout(dt)

    model = MTBERTClassification.from_pretrained(conf.bert,
                                                 id2label_a=handlerA.id2label,
                                                 id2label_b=handlerB.id2label)

    if conf.cuda:
        model = model.to(conf.gpu)

    train(model, handlerA, handlerB, df_train, df_val, conf)
