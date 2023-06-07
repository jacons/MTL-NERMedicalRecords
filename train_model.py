from Configuration import Configuration
from Parsing.parser_utils import parse_args, buildDataset, holdout
from Training.MTModel import Classifier
from Training.Trainer import train

if __name__ == '__main__':

    args, _ = parse_args()

    conf = Configuration(args)
    conf.show_parameters(["param", "bert"])

    if args.model_name is None:
        raise Exception("Define a model name!")

    dt, handlerA, handlerB = buildDataset(args.datasets, verbose=True)

    df_train, df_val, _ = holdout(dt)

    model = Classifier(conf.bert, len(handlerA.set_entities), len(handlerB.set_entities))

    if conf.cuda:
        model = model.to(conf.gpu)

    train(model, handlerA, handlerB, df_train[:10000], df_val[:10000], conf)
