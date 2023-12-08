class ArgObj(object):
    _defaults = {
        "max_epochs": 50,
        "patience": 30,
        "device": "cuda",
        "seed": 42,
        "lr": 0.01,
        "momentum": 0.7,
        "weight_decay": 0.0005,
        "batch_size": 32,
        "n_classes": 100,
        "dropout": 0.5,
        "log_interval": 10,
        "loss_limit": 0.01
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)