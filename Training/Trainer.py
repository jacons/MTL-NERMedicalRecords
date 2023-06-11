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
          handler_a: EntityHandler,
          handler_b: EntityHandler,
          df_train: DataFrame,
          df_val: DataFrame,
          conf: Configuration):
    """

    :param model: MultiHead Model
    :param handler_a: Handler for the labels a
    :param handler_b: Handler for the labels b
    :param df_train: training dataset
    :param df_val: validation dataset
    :param conf: configuration parameters
    :return:
    """

    # --------- DATASETS ---------
    print("--INFO--\tCreating Dataloader for Training set")
    tr = DataLoader(NerDataset(df_train, conf, handler_a, handler_b), collate_fn=padding_batch,
                    batch_size=conf.param["batch_size"], shuffle=True)

    print("\n--INFO--\tCreating Dataloader for Validation set")
    vl = DataLoader(NerDataset(df_val, conf, handler_a, handler_b))
    # --------- DATASETS ---------

    epoch = 0
    tr_size, vl_size = len(tr), len(vl)
    total_epochs = conf.param["max_epoch"]
    stopping = conf.param["early_stopping"]  # "Patience in early stopping"

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
    while (epoch < total_epochs) and (not es.earlyStop):

        loss_train, loss_val = 0, 0
        confusion_a = zeros(size=(len(handler_a), len(handler_a)))
        confusion_b = zeros(size=(len(handler_b), len(handler_b)))

        # ========== Training Phase ==========
        model.train()
        #  There inputs are created in "NerDataset" class
        for inputs_ids, att_mask, _, _, labels_a, labels_b in tqdm(tr, desc="Training", mininterval=conf.refresh_rate):
            optimizer.zero_grad(set_to_none=True)

            loss, _, _ = model(inputs_ids, att_mask, labels_a, labels_b)
            loss_train += loss.item()

            loss.backward()
            optimizer.step()
        # ========== Training Phase ==========

        # ========== Validation Phase ==========
        model.eval()
        with no_grad():  # Validation phase
            for fields in tqdm(vl, desc="Evaluation", mininterval=conf.refresh_rate):
                inputs_ids, att_mask, tag_mask_a, tag_mask_b, labels_a, labels_b = fields

                loss, path_a, path_b = model(inputs_ids, att_mask, labels_a, labels_b)
                loss_val += loss.item()

                if conf.cuda:
                    path_a = path_a.to(conf.gpu)
                    path_b = path_b.to(conf.gpu)

                labels_a = masked_select(labels_a, tag_mask_a)
                path_a = masked_select(path_a, tag_mask_a)

                for lbl, pre in zip(labels_a, path_a):
                    confusion_a[lbl, pre] += 1

                labels_b = masked_select(labels_b, tag_mask_b)
                path_b = masked_select(path_b, tag_mask_b)

                for lbl, pre in zip(labels_b, path_b):
                    confusion_b[lbl, pre] += 1

        # ========== Validation Phase ==========

        tr_loss, val_loss = (loss_train / tr_size), (loss_val / vl_size)
        f1_scores = scores(confusion_a), scores(confusion_b)

        print(f'Epochs: {epoch + 1} | Loss: {tr_loss: .4f} | Val_Loss: {val_loss: .4f} | '
              f'F1: {f1_scores[0]: .4f} {f1_scores[1]: .4f}')

        epoch += 1
        f1_score = 0.5 * (f1_scores[0] + f1_scores[1])

        if model_version is not None:
            # save the model if it is the best model until now
            model_version.update(model, f1_score * -1)

        # Update the scheduler
        scheduler.step(val_loss)
        # Update the early stopping
        es.update(f1_score * -1)
