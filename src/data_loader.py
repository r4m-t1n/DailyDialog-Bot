import re
import os
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

BASE_DIR = r"..\data"
TRAIN_DIR = os.path.join(BASE_DIR, r"train\dialogues_train.txt")
VALID_DIR = os.path.join(BASE_DIR, r"validation\dialogues_validation.txt")
TEST_DIR = os.path.join(BASE_DIR, r"test\dialogues_test.txt")

class Seq2SeqDataset(Dataset):
    def __init__(self, df):
        self.contexts = df["context_idx_padded"].tolist()
        self.responses = df["response_idx_padded"].tolist()

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        context = torch.tensor(self.contexts[idx], dtype=torch.long)
        response = torch.tensor(self.responses[idx], dtype=torch.long)
        return context, response

def load_dialogue_pairs(file_path):
    with open(file_path, "r", encoding="UTF-8") as f:
        raw_lines = f.readlines()

    dialogs = [line.strip().split("__eou__") for line in raw_lines]
    dialogs_cleaned = [[utt.strip() for utt in dialog if utt.strip()] for dialog in dialogs]

    pairs = []
    for dialog in dialogs_cleaned:
        for i in range(len(dialog) - 1):
            context = " ".join(dialog[:i+1]).strip()
            response = dialog[i+1].strip()
            if context and response:
                pairs.append({"context": context, "response": response})
    return pairs

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

def pad(seq, max_len, pad_value=0):
    if len(seq) > max_len:
        seq = seq[:max_len]
    elif len(seq) < max_len:
        seq = seq + [pad_value] * (max_len - len(seq))
    return seq

train_pairs = load_dialogue_pairs(TRAIN_DIR)
valid_pairs = load_dialogue_pairs(VALID_DIR)
test_pairs = load_dialogue_pairs(TEST_DIR)

df_train = pd.DataFrame(train_pairs)
df_valid = pd.DataFrame(valid_pairs)
df_test = pd.DataFrame(test_pairs)

for df in [df_train, df_valid, df_test]:
    df["context_clean"] = df["context"].apply(preprocess_text)
    df["response_clean"] = df["response"].apply(preprocess_text)
    df["context_tokens"] = df["context_clean"].apply(lambda x: x.split())
    df["response_tokens"] = df["response_clean"].apply(lambda x: x.split())

all_tokens = []
for df in [df_train, df_valid, df_test]:
    for tokens_list in df["context_tokens"]:
        all_tokens.extend(tokens_list)
    for tokens_list in df["response_tokens"]:
        all_tokens.extend(tokens_list)

word_counts = Counter(all_tokens)

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

vocab = {
    word: i + 4 for i, (word, _) in enumerate(word_counts.most_common()) if word not in ["<PAD>", "<UNK>", "<SOS>", "<EOS>"]
}
vocab["<PAD>"] = PAD_IDX
vocab["<UNK>"] = UNK_IDX
vocab["<SOS>"] = SOS_IDX
vocab["<EOS>"] = EOS_IDX

idx2word = {idx: word for word, idx in vocab.items()}

max_len_context = 40
max_len_response = 42

for df in [df_train, df_valid, df_test]:
    df["context_idx"] = df["context_tokens"].apply(
        lambda x: tokens_to_indices(x, vocab)
    )
    df["response_idx"] = df["response_tokens"].apply(
        lambda x: [vocab["<SOS>"]] + tokens_to_indices(x, vocab) + [vocab["<EOS>"]]
    )
    df["context_idx_padded"] = df["context_idx"].apply(
        lambda x: pad(x, max_len_context, PAD_IDX)
    )
    df["response_idx_padded"] = df["response_idx"].apply(
        lambda x: pad(x, max_len_response, PAD_IDX)
    )

BATCH_SIZE = 32

train_dataset = Seq2SeqDataset(df_train)
valid_dataset = Seq2SeqDataset(df_valid)
test_dataset = Seq2SeqDataset(df_test)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)