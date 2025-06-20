NOTE_CONFIG = {
    'iabn': True,
    'alpha': 4,
    'bn_momentum': 0.1,
    'use_learned_stats': True,
    'save_path': None
}

CIFAR10_CONFIG = {
    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
}