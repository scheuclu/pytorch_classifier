from local_configs.train_config import TrainConfig
import local_configs.numclass_5_configs as numclass_5_configs
import local_configs.numclass_5_other_configs as numclass_5_other_configs
import local_configs.numclass_5_other_random_configs as numclass_5_other_random_configs


debug = {
    'train_identifier': 'debug1',
    'imageset': 'imageset_5',
    'model': 'ResNet18',
    'cross_val_phase': 0,
    'batchsize': 500,
    'cross_val_phase': 1,
    'port': 6065,
    'optimizer': {
        'lr': 0.1,
        'momentum': 0.9
    },
    'lr_scheduler': {
        'type': 'StepLR',
        'step_size': 10,
        'gamma': 0.5
    }
}

configs={
    "debug": TrainConfig(debug),
    "long_train_0" : TrainConfig(numclass_5_configs.long_train_0),
    "long_train_1" : TrainConfig(numclass_5_configs.long_train_1),
    "resnet_set_0" : TrainConfig(numclass_5_configs.resnet_set_0),
    "resnet_set_1" : TrainConfig(numclass_5_configs.resnet_set_1),
    "densenet169_set_0" : TrainConfig(numclass_5_configs.densenet169_set_0),
    "densenet169_set_1" : TrainConfig(numclass_5_configs.densenet169_set_1),

    "squeezenet_0_other": TrainConfig(numclass_5_other_configs.squeezenet_0_withother),
    "squeezenet_1_other": TrainConfig(numclass_5_other_configs.squeezenet_1_withother),
    "resnet_set_0_other": TrainConfig(numclass_5_other_configs.resnet_set_0_withother),
    "resnet_set_1_other": TrainConfig(numclass_5_other_configs.resnet_set_1_withother),
    "densenet169_set_0_other": TrainConfig(numclass_5_other_configs.densenet169_set_0_withother),
    "densenet169_set_1_other": TrainConfig(numclass_5_other_configs.densenet169_set_1_withother),

    "squeezenet_0_other_random": TrainConfig(numclass_5_other_random_configs.squeezenet_0_other_random),
    "squeezenet_1_other_random": TrainConfig(numclass_5_other_random_configs.squeezenet_1_other_random),
    "resnet_set_0_other_random": TrainConfig(numclass_5_other_random_configs.resnet_set_0_other_random),
    "resnet_set_1_other_random": TrainConfig(numclass_5_other_random_configs.resnet_set_1_other_random),
    "densenet169_set_0_other_random": TrainConfig(numclass_5_other_random_configs.densenet169_set_0_other_random),
    "densenet169_set_1_other_random": TrainConfig(numclass_5_other_random_configs.densenet169_set_1_other_random),
}

