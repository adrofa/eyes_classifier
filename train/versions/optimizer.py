import torch


def get_optimizer(version, model_parameters, weights=None):
    if version == "adam_1":
        # lr found via torch_lr_finder: lr_finder/v1.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=3.20E-02)
        if weights:
            optimizer.load_state_dict(torch.load(weights))

    elif version == "adam_2":
        # lr found via torch_lr_finder: lr_finder/v2.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=4.53E-03)
        if weights:
            optimizer.load_state_dict(torch.load(weights))

    elif version == "adam_3":
        # lr found via torch_lr_finder: lr_finder/v3.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=3.43E-03)
        if weights:
            optimizer.load_state_dict(torch.load(weights))

    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")
    return optimizer
