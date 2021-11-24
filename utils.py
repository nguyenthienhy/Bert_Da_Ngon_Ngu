from tqdm import tqdm
tqdm.pandas()
import numpy as np
import pickle
import torch
import pickle
from torch.utils.data import Dataset
from transformers import BertTokenizer

class Tokenizer(Dataset):
  def __init__(self, bert_type=None):
    self.bertTokenizer = BertTokenizer.from_pretrained(bert_type)

def convert_lines(df, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    for idx, row in enumerate(df): 
        tokens = bpe.tokenize(row)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        # Đưa về cùng độ dài: Cắt đi nếu dài hơn max_seq_len/Thêm pad nếu ngắn hơn max_seq_len 
        if len(tokens) < max_sequence_length: 
            tokens = tokens + ['[PAD]' for i in range(max_sequence_length - len(tokens))]
        else: 
            tokens = tokens[: (max_sequence_length-1)] + ['[SEP]']
        # Convert to ids
        token_ids = bpe.convert_tokens_to_ids(tokens)
        outputs[idx,:] = np.array(token_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def save_to_pickle_file(object , path):
    with open(path , 'wb') as f:
        pickle.dump(object , f)

def load_pickle_file(path):
    with open(path , 'rb') as f:
        return pickle.load(f)