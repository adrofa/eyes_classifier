import torch


def get_optimizer(version, model_parameters, weights=None):
    if version == "adam_1":
        # lr found via torch_lr_finder: lr_finder/v1.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=1.52E-02)

        if weights:
            optimizer.load_state_dict(torch.load(weights))
    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")
    return optimizer
