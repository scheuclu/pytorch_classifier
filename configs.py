


configs={

    'debug':{
        'train_identifier': 'debug',
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
    }

}