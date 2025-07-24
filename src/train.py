import torch
import torch.nn as nn
import torch.optim as optim
import time

from model import Encoder, Decoder, Seq2Seq
from data_loader import train_loader, vocab, PAD_IDX, valid_loader

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0
    
    for i, (src, trg) in enumerate(iterator):

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=0.5)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src = src.to(device)
            trg = trg.to(device)

            output = model(src, trg, teacher_forcing_ratio=0) 

            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

LEN_VOCAB = len(vocab)
EMB_DIM = 128
EMB_DIM = 128
HID_DIM = 256
N_LAYERS = 1

encoder = Encoder(LEN_VOCAB, EMB_DIM, HID_DIM, N_LAYERS).to(device)
decoder = Decoder(LEN_VOCAB, EMB_DIM, HID_DIM, N_LAYERS).to(device)
model = Seq2Seq(encoder, decoder, device).to(device)

learning_rate = 0.001
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')
MODEL_SAVE_PATH = 'dd_chatbot_model.pt'

if __name__ == "__main__":
    for epoch in range(N_EPOCHS):
        start_time = time.time()

        train_loss = train(model, train_loader, optimizer, criterion, CLIP)
        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f}')
        print(f'\tVal Loss: {valid_loss:.3f}')

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Model is saved with Val Loss: {valid_loss:.3f}")