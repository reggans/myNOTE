NOTE_CONFIG = {
    'train_config': {
        "method": "NOTE",
        "lr": 1e-3,
        "momentum": 0.1,
        "weight_decay": 5e-4,
        "epochs": 1,
        "batch_size": 64,
    },
    'online_config':{
        "use_learned_stats": True,
        "update_interval": 64,
        "temp_factor": 1.0,
        "optimize": True,
    },
    'save_path': 'logs/NOTE',
    'iabn': True,
    'alpha': 4,
    'bn_momentum': 0.1,
    'memory_type': "PBRS",
    'capacity': 64,

}

# Config for CIFAR10, mostly for target domain
CIFAR10_CONFIG = {
    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    'domains': ['test'],
    'distribution': 'random',
    'dir_beta': 0.1,
}