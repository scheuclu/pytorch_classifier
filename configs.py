
classnames = ['Car', 'Pedestrian', 'Cyclist', 'TrafficSign', 'TrafficSignal']

name2index = {'Car': 0, 'Pedestrian': 1, 'Cyclist': 2, 'TrafficSign': 3, 'TrafficSignal': 4}

index2name = { val:key for key, val in name2index.items()}


configs={

    'debug':{
        'train_identifier': 'debug',
        'model': 'DenseNet161',
        'port': 6065,
        'optimizer': {
            'lr': 0.1,
            'momentum': 0.9
        },
        'lr_scheduler':{
            'type': 'StepLR',
            'step_size': 7,
            'gamma': 0.1
        }
    },
    'long_train_0': {
        'train_identifier': 'long_train_0',
        'model': 'SqueezeNet',
        'port': 6065,
        'optimizer': {
            'lr': 10.0,
            'momentum': 0.9
        },
        'lr_scheduler': {
            'type': 'StepLR',
            'step_size': 7,
            'gamma': 0.1
        }
    },
    'long_train_1': {
        'train_identifier': 'long_train_1',
        'model': 'SqueezeNet',
        'port': 6065,
        'optimizer': {
            'lr': 0.1,
            'momentum': 0.9
        },
        'lr_scheduler': {
            'type': 'StepLR',
            'step_size': 7,
            'gamma': 0.5
        }
    }

}