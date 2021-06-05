import albumentations as A


def get_augmentation(version):
    if version == 1:
        # for details see: /scripts/image_normalization.py
        normalization = A.Normalize(mean=0.499, std=0.198, p=1)

        augmentation = {
            "train": A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),

                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.25),
                A.Blur(blur_limit=4, always_apply=False, p=0.25),

                normalization
            ]),

            "valid": normalization
        }

    else:
        raise Exception(f"Augmentation version '{version}' is unknown!")
    return augmentation
