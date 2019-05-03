squeezenet_0_withother = {
    'train_identifier': 'squeezenet_0_withother',
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

squeezenet_1_withother = {
    'train_identifier': 'squeezenet_1_withother',
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

resnet_set_0_withother = {
    'train_identifier': 'resnet_set_0_withother',
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

resnet_set_1_withother = {
    'train_identifier': 'resnet_set_1_withother',
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

densenet169_set_0_withother = {
    'train_identifier': 'densenet169_set_0_withother',
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
        'step_size': 15,
        'gamma': 0.5
    }
}

densenet169_set_1_withother = {
    'train_identifier': 'densenet169_set_1_withother',
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
        'step_size': 15,
        'gamma': 0.5
    }
}
