import albumentations as A


def get_augmentation(version):
    if version == 1:
        # for details see: /utils/image_normalization.py
        normalization = A.Normalize(mean=0.499, std=0.198, p=1)
        augmentation = {
            "train": A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.25),
                A.Blur(blur_limit=4, always_apply=False, p=0.25),
                normalization
            ]),
            "valid": normalization
        }
    elif version == 2:
        # for details see: /utils/image_normalization.py
        normalization = A.Normalize(mean=0.499, std=0.198, p=1)
        augmentation = {
            "train": A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5, p=0.25),
                A.Blur(blur_limit=5, always_apply=False, p=0.25),

                A.OneOf([
                    A.ElasticTransform(p=1, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03,),
                    A.GridDistortion(p=1),
                ], p=0.25),

                normalization
            ]),
            "valid": normalization
        }
    else:
        raise Exception(f"Augmentation version '{version}' is unknown!")
    return augmentation
