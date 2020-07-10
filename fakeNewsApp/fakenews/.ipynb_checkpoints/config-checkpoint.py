import torch
from transformers import BertTokenizer

SEED = 42
MAX_LEN = 512
TRAIN_BATCH_SIZE = 2
VALID_BATCH_SIZE = 2
EPOCHS = 3
DROPOUT = 0.3
LEARNING_RATE = 4e-5
TRAINING_STAT_PATH = "output/"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BERT_TOKENIZER_PATH = "bert-base-uncased"
BERT_MODEL_PATH = "bert-base-uncased"

BERT_MODEL_OUTPUT = "output/checkpoint.pt"

DATA_PATH = "data/"

LOG_DIR = "output/log/"
OUTPUT_DIR = "output/"
MODEL_CHECKPOINT = "output/model.pt"
