from pandas import DataFrame
from torch import no_grad, zeros, masked_select
from torch.nn import Module
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.sgd import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm

import Configuration
from Evaluation.metrics import scores
from Parsing.CustomDataset import NerDataset
from Parsing.parser_utils import EntityHandler
from Training.trainer_utils import padding_batch, EarlyStopping, ModelVersion


def train(model: Module,
          e_handler_a: EntityHandler,
          e_handler_b: EntityHandler,
          df_train: DataFrame,
          df_val: DataFrame,
          conf: Configuration):
    """

    :param model:
    :param e_handler_a:
    :param e_handler_b:
    :param df_train:
    :param df_val:
    :param conf:
    :return:
    """

    # --------- DATASETS ---------
    print("--INFO--\tCreating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, conf, e_handler_a, e_handler_b), collate_fn=padding_batch,
                    batch_size=conf.param["batch_size"], shuffle=True)

    print("\n--INFO--\tCreating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, conf, e_handler_a, e_handler_b))
    # --------- DATASETS ---------

    epoch = 0
    tr_size, vl_size = len(tr), len(vl)
    total_epochs = conf.param["max_epoch"]
    stopping = conf.param["early_stopping"]  # "Patience in early stopping"

    max_labels_a = len(e_handler_a.set_entities)
    max_labels_b = len(e_handler_b.set_entities)

    # --------- Early stopping ---------
    es = EarlyStopping(total_epochs if stopping <= 0 else stopping)

    # --------- Optimizer ---------
    optimizer = SGD(model.parameters(), lr=conf.param["lr"], momentum=conf.param["momentum"],
                    weight_decay=conf.param["weight_decay"], nesterov=True)

    # --------- Save only the best model (which have minimum validation loss) ---------
    model_version = ModelVersion(folder=conf.folder, name=conf.model_name) if conf.save_model else None

    # --------- Scheduling the learning rate to improve the convergence ---------
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3)

    print("\n--INFO--\tThe Training is started")
    model.train()
    while (epoch < total_epochs) and (not es.earlyStop):

        loss_train, loss_val = 0, 0

        # ========== Training Phase ==========

        #  There inputs are created in "NerDataset" class
        for inputs_ids, att_mask, _, labels_a, labels_b in tqdm(tr, desc="Training", mininterval=conf.refresh_rate):
            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = model(inputs_ids, att_mask, labels_a, labels_b)
            loss_train += loss.item()

            loss.backward()
            optimizer.step()
        # ========== Training Phase ==========

        # ========== Validation Phase ==========
        confusion_a = zeros(size=(max_labels_a, max_labels_a))
        confusion_b = zeros(size=(max_labels_b, max_labels_b))

        with no_grad():  # Validation phase
            for inputs_ids, att_mask, tag_maks, labels_a, labels_b in tqdm(vl, desc="Evaluation",
                                                                           mininterval=conf.refresh_rate):
                loss, logits_a, logits_b = model(inputs_ids, att_mask, labels_a, labels_b)
                loss_val += loss.item()

                # ----- CRF Version -----
                # path, _ = logits[0]
                # path = torch.LongTensor(path)

                # if conf.cuda:
                #    path = path.to(conf.gpu)
                # logits = masked_select(path, tag_maks)
                # labels = masked_select(labels, tag_maks)
                # for lbl, pre in zip(labels, logits):
                #   confusion[lbl, pre] += 1
                # ----- CRF Version -----

                logits_a = logits_a[0].argmax(1)  # label predicted
                logits_b = logits_b[0].argmax(1)  # label predicted

                labels_a = masked_select(labels_a, tag_maks)
                labels_b = masked_select(labels_b, tag_maks)

                logits_a = masked_select(logits_a, tag_maks)
                logits_b = masked_select(logits_b, tag_maks)

                for lbl, pre in zip(labels_a, logits_a):
                    confusion_a[lbl, pre] += 1

                for lbl, pre in zip(labels_b, logits_b):
                    confusion_b[lbl, pre] += 1

        # ========== Validation Phase ==========
        model.train()

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)
        f1_score = (scores(confusion_a) + scores(confusion_b)) / 2

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .4f} | Val_Loss: {val_loss: .4f} | F1: {f1_score: .4f}')
        epoch += 1

        if model_version is not None:
            # save the model if it is the best model until now
            model_version.update(model, f1_score * -1)

        # Update the scheduler
        scheduler.step(val_loss)
        # Update the early stopping
        es.update(f1_score * -1)
