import torch


def get_optimizer(version, model_parameters, weights=None):
    if version == "adam_1":
        # lr = Karpathy Score
        optimizer = torch.optim.Adam(params=model_parameters, lr=0.01)
        if weights:
            optimizer.load_state_dict(torch.load(weights))
    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")
    return optimizer
