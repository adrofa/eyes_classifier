import torch


def get_optimizer(version, model_parameters, weights=None):
    if version == "adam_1":
        # lr found via torch_lr_finder: lr_finder/v1.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=3.20E-02)

    elif version == "adam_2":
        # lr found via torch_lr_finder: lr_finder/v2.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=4.53E-03)

    elif version == "adam_3":
        # lr found via torch_lr_finder: lr_finder/v3.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=3.43E-03)

    elif version == "adam_4":
        # lr found via torch_lr_finder: lr_finder/v4.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=1.79E-03)

    elif version == "adam_5":
        # lr found via torch_lr_finder: lr_finder/v5.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=2.36E-03)

    elif version == "adam_6":
        # lr found via torch_lr_finder: lr_finder/v6.ipynb
        optimizer = torch.optim.Adam(params=model_parameters, lr=2.15E-03)

    else:
        raise Exception(f"Optimizer version '{version}' is unknown!")

    if weights:
        optimizer.load_state_dict(torch.load(weights))

    return optimizer
