import torch
CHECKPOINT_DIR = "checkpoints"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = "different_datasets/male_ai_generated"
WOMEN="different_datasets/woman_ai"
BATCH_SIZE = 8
NUM_WORKERS = 2
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 60
SAVE_PATH = "checkpoint.pt"
LOG_PATH = "benchmark.csv"

LOAD=False
LOAD_PATH="checkpoints/epoch_2.pt"