NOTE_CONFIG = {
    'train_config': {
        "method": "NOTE",
        # Dataset-specific
        "lr": None,
        "momentum": None,
        "weight_decay": None,
        "batch_size": None,
        "epochs": None,
    },
    'online_config':{
        "epochs": 1,
        "use_learned_stats": True,
        "update_interval": 64,
        "temp_factor": 1.0,
        "optimize": True,
    },
    'save_path': 'logs/NOTE/',
    'checkpoint_path': 'ckpt/NOTE/',
    'iabn': True,
    'alpha': 4,
    'bn_momentum': 0.01,
    'memory_type': "PBRS",
    'capacity': 64,

}

# Config for CIFAR10, mostly for target domain
CIFAR10_CONFIG = {
    'learning_rate': 0.1, #initial learning rate
    'weight_decay': 0.0005,
    'momentum': 0.9,
    'batch_size': 128,
    'epochs': 200,
    'img_size': 3072,

    'file_path': './dataset/CIFAR-10-C',
    'classes': ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'],
    'num_class': 10,
    'severity': 5,
    'domains': ['test'],
    'distribution': 'dirichlet',
    'dir_beta': 0.1,
}