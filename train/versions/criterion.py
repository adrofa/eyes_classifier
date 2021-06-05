import torch.nn as nn


def get_criterion(version):
    if version == 1:
        criterion = nn.BCEWithLogitsLoss(reduction="mean")
    else:
        raise Exception(f"Criterion version '{version}' is unknown!")
    return criterion