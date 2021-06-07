import torch


def get_scheduler(version, optimizer):
    if version == "rop_1":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.1, patience=5, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )

    elif version == "rop_2":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min', factor=0.1, patience=3, threshold=0.0001,
            threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=True
        )

    elif version == "clc_1":
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer, base_lr=0.0001, max_lr=0.035, step_size_up=1, step_size_down=10,
            mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle',
            cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1
        )

    else:
        raise Exception(f"Scheduler version '{version}' is unknown!")
    return scheduler
