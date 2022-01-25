def get_augmentations_from_config(self, augmentations):
    if hasattr(augmentations, "FROM_DICT"):
        if augmentations.FROM_DICT is not None:
            return A.from_dict(OmegaConf.to_container(augmentations.FROM_DICT))
    transforms = list(augmentations.keys())
    trans = []
    for transform in transforms:
        parameters = getattr(augmentations, transform)
        if parameters == None: parameters = {}
        try:
            # try to load the functions from ALbumentations(A)
            func = getattr(A, transform)
            trans.append(func(**parameters))
        except:
            try:
                # exeption for ToTensorV2 function which is in A.pytorch
                func = getattr(A.pytorch, transform)
                trans.append(func(**parameters))
            except:
                print("No Operation Found", transform)
    return A.Compose(trans)