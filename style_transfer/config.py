import os

DEFAULT_CONFIG = {
    "DATASET_PATH": "/kaggle/input/style-transfer/custom_coco_dataset",
    "TRAIN_CONTENT_DIR": None,
    "TRAIN_STYLE_DIR": None,
    "TEST_CONTENT_DIR": None,
    "TEST_STYLE_DIR": None,

    "IMAGE_SIZE": 512,
    "CROP_SIZE": 256,
    "BATCH_SIZE": 8,

    "EPOCHS": 12,
    "LAMBDA_STYLE": 12.0,
    "LEARNING_RATE": 1e-4,
    "SAVE_PATH": "/kaggle/working/models",
}

def get_config(override_dict: dict = None) -> dict:
    config = DEFAULT_CONFIG.copy()

    if override_dict:
        config.update({k: override_dict[k] for k in override_dict if override_dict[k] is not None})

    dataset_path = config["DATASET_PATH"]
    if config["TRAIN_CONTENT_DIR"] is None:
        config["TRAIN_CONTENT_DIR"] = os.path.join(dataset_path, "train2017")
    if config["TEST_CONTENT_DIR"] is None:
        config["TEST_CONTENT_DIR"] = os.path.join(dataset_path, "val2017")
    if config["TRAIN_STYLE_DIR"] is None:
        config["TRAIN_STYLE_DIR"] = os.path.join(dataset_path, "painter_train")
    if config["TEST_STYLE_DIR"] is None:
        config["TEST_STYLE_DIR"] = os.path.join(dataset_path, "painter_val")

    return config
