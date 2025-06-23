if __name__ == "__main__":
    train_config = {
        "method": "NOTE",
        "lr": 1e-3,
        "momentum": 0.1,
        "weight_decay": 5e-4,
        "epochs": 100,
        "batch_size": 64,
    }
    online_config = {
        "use_learned_stats": True,
        "update_interval": 64,
        "temp_factor": 1.0,
    }