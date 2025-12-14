# Training hyperparameters - base model (TinyNet)
EPOCHS_base = 20
BATCH_SIZE_base = 32
LEARNING_RATE_base = 1e-4

# Training hyperparameters - resnet18-based model
EPOCHS_resnet18 = 20
BATCH_SIZE_resnet18 = 16
LEARNING_RATE_resnet18 = 1e-3

# Paths
TRAIN_CSV = "/data/split/train.csv"
VAL_CSV = "/data/split/val.csv"
BASE_FINAL_PATH = "/data/tinynet_final.pth"
RES_FINAL_PATH = "/data/resnet18_final.pth"
BASE_BEST_PATH = "/data/tinynet_best.pth"
RES_BEST_PATH = "/data/resnet18_best.pth"
CONSENSUS_LABELS_CSV = "/data/consensus_label_distribution.csv"
DATA_DIR = "/data"
ZIP_PATH = "/data/raw.zip"
PREPARED_CSV = "/data/prepared_dataset.csv"
CONSENSUS_TEST_CSV = "/data/split/consensus_test.csv"
