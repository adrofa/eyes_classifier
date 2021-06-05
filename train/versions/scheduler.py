import torch


def get_scheduler(version, optimizer):
    if version == "rop_1":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.1, patience=10, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )
    else:
        raise Exception(f"Scheduler version '{version}' is unknown!")
    return scheduler
