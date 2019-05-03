
def ensure_dict_has_keys(dictname, d, names):
    for name in names:
        if name not in d:
            raise ValueError(dictname,"is missing key:", name)
    for key in d.keys():
        if key not in names:
            raise ValueError(dictname, "got unknown key:", key)
    pass


class OptimizerConfig(object):
    def __init__(self, optimizer_config):
        ensure_dict_has_keys("Optimizer", optimizer_config, ["lr", "momentum"])
        self.lr = optimizer_config["lr"]
        self.momentum = optimizer_config["momentum"]

class LRSchedulerConfig(object):
    def __init__(self, lrscheduler_config):
        ensure_dict_has_keys("LR Scheduler", lrscheduler_config, ["type", "step_size", "gamma"])
        self.type = lrscheduler_config["type"]
        self.step_size = lrscheduler_config["step_size"]
        self.gamma = lrscheduler_config["gamma"]


class TrainConfig(object):
    def __init__(self, train_config):
        ensure_dict_has_keys("Train", train_config,
            ["train_identifier", "model", "port", "optimizer", "lr_scheduler", "cross_val_phase", "batchsize"])
        self.train_identifier = train_config["train_identifier"]
        self.model = train_config["model"]
        self.batchsize = train_config["batchsize"]
        self.port = train_config["port"]
        self.cross_val_phase = train_config["cross_val_phase"]
        self.optimizer = OptimizerConfig(train_config["optimizer"])
        self.lr_scheduler = LRSchedulerConfig(train_config["lr_scheduler"])


        # 'long_train_1': {
        #                     'train_identifier': 'long_train_1',
        #                     'model': 'ResNet18',
        #                     'port': 6065,
        #                     'optimizer': {
        #                         'lr': 0.1,
        #                         'momentum': 0.9
        #                     },
        #                     'lr_scheduler': {
        #                         'type': 'StepLR',
        #                         'step_size': 10,
        #                         'gamma': 0.5
        #                     }
        #                 },