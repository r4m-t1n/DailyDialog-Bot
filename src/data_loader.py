import re
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader

TRAIN_DIR = r"..\data\EMNLP_dataset\train\dialogues_train.txt"

with open(TRAIN_DIR, "r", encoding="UTF-8") as f:
    raw_lines = f.readlines()

dialogs = [line.strip().split("__eou__") for line in raw_lines]
dialogs_cleaned = [[utt.strip() for utt in dialog if utt.strip()] for dialog in dialogs]

pairs = []
for dialog in dialogs_cleaned:
    for i in range(len(dialog) - 1):
        context = " ".join(dialog[:i+1]).strip()
        response = dialog[i+1].strip()
        if context and response:
            pairs.append((context, response))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()

context_tokens = [preprocess_text(c).split() for c, _ in pairs]
response_tokens = [preprocess_text(r).split() for _, r in pairs]
all_tokens = [tok for pair in context_tokens + response_tokens for tok in pair]

word_counts = Counter(all_tokens)
vocab = {word: i + 2 for i, (word, _) in enumerate(word_counts.most_common())}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

def tokens_to_indices(tokens):
    return [vocab.get(tok, vocab["<UNK>"]) for tok in tokens]

context_indices = [tokens_to_indices(tokens) for tokens in context_tokens]
response_indices = [tokens_to_indices(tokens) for tokens in response_tokens]

def pad(seq, max_len, pad_value=0):
    return seq[:max_len] + [pad_value] * (max_len - len(seq))

max_len_context = 40
max_len_response = 40

context_padded = [pad(seq, max_len_context) for seq in context_indices]
response_padded = [pad(seq, max_len_response) for seq in response_indices]

class Seq2SeqDataset(Dataset):
    def __init__(self, context_seqs, response_seqs):
        self.contexts = context_seqs
        self.responses = response_seqs

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.contexts[idx], dtype=torch.long),
            torch.tensor(self.responses[idx], dtype=torch.long)
        )

dataset = Seq2SeqDataset(context_padded, response_padded)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
