long_train_0 = {
    'train_identifier': 'long_train_0',
    'cross_val_phase': 0,
    'model': 'SqueezeNet',
    'batchsize': 100,
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

long_train_1 = {
    'train_identifier': 'long_train_1',
    'cross_val_phase': 1,
    'model': 'ResNet18',
    'batchsize': 100,
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

resnet_set_0 = {
    'train_identifier': 'resnet_set_0',
    'cross_val_phase': 0,
    'model': 'ResNet18',
    'batchsize': 300,
    'cross_val_phase': 0,
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

resnet_set_1 = {
    'train_identifier': 'resnet_set_1',
    'cross_val_phase': 1,
    'model': 'ResNet18',
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

densenet169_set_0 = {
    'train_identifier': 'densenet169_set_0',
    'model': 'DenseNet169',
    'batchsize': 80,
    'cross_val_phase': 0,
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

densenet169_set_1 = {
    'train_identifier': 'densenet169_set_1',
    'model': 'DenseNet169',
    'batchsize': 80,
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
