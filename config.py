class ArgObj(object):
    _defaults = {
        "max_epochs": 50,
        "patience": 5,
        "device": "cuda",
        "lr": 0.01,
        "dataset_seed": 42,
        "momentum": 0.7,
        "weight_decay": 0.0005,
        "batch_size": 64,
        "n_classes": 100,
        "dropout": 0.5,
        "log_interval": 10,
        "seedlist": [751,456,894,564,483]
    }

    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        self.__dict__.update(kwargs)