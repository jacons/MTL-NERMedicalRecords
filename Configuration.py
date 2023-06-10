import torch


class Configuration:
    """
    Class used to store all parameters and configuration for the execution
    """

    def __init__(self, p):

        # Hyperparameters
        self.param: dict = {
            "lr": p.lr,
            "momentum": p.momentum,
            "weight_decay": p.weight_decay,
            "batch_size": p.batch_size,
            "max_epoch": p.max_epoch,
            "early_stopping": p.patience,
        }

        self.save_model = True if p.save == 1 else False
        self.bert = p.bert  # Bert model as baseline

        self.model_name = p.model_name
        self.folder = p.path_model  # Directory to save the model

        # The system recognizes if there are some GPU available
        self.cuda = True if torch.cuda.is_available() else False
        self.gpu = "cuda:0"

        self.refresh_rate: int = p.refresh_rate  # interval of refresh in tqdm

    def update_params(self, param: str, value: float):
        self.param[param] = value

    def show_parameters(self, conf=None) -> None:
        if conf is None:
            conf = []

        if "bert" in conf:
            print("{:<85}".format("Bert model"))
            print("-" * 85)
            print("|{:^83}|".format(self.bert))
            print("-" * 85)

        if "param" in conf:
            print("{:<85}".format("Parameters & Values"))
            print("-" * 85)
            param_items = list(self.param.items())
            num_params = len(param_items)
            for idx in range(0, num_params, 3):
                param_slice = param_items[idx:idx + 3]
                param_row = "|".join("{:^14} {:^12}".format(k, v) for k, v in param_slice)
                print(param_row)
                print("-" * 85)

        return
