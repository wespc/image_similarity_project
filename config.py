def get_config(dataset):
    if dataset == 'mnist':
        return {
            'in_channels': 1,
            'batch_size': 64,
            'lr': 0.001,
            'epochs': 10,
            'transform': 'mnist',
            'model_type': 'simple'
        }
    elif dataset == 'cifar':
        return {
            'in_channels': 3,
            'batch_size': 32,
            'lr': 0.0005,
            'epochs': 20,
            'transform': 'cifar',
            'model_type': 'enhanced'
        }
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")