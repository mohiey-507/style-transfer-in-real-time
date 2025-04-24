import os 

DATASET_PATH = '/kaggle/input/style-transfer/custom_coco_dataset'
TRAIN_CONTENT_DIR = os.path.join(DATASET_PATH, 'train2017')
TEST_CONTENT_DIR = os.path.join(DATASET_PATH, 'val2017')
TRAIN_STYLE_DIR = os.path.join(DATASET_PATH, 'painter_train')
TEST_STYLE_DIR = os.path.join(DATASET_PATH, 'painter_val')


IMAGE_SIZE = 512
CROP_SIZE = 256
BATCH_SIZE = 8
NUM_WORKERS = 4
PIN_MEMORY = True
DROP_LAST = True