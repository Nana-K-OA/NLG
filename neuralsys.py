import sys
import csv
import re
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import Counter
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

# Hyperparameters i constantly adjusted
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EMBED_SIZE = 128
NUM_LAYERS = 2
NUM_HEADS = 4
LEARNING_RATE = 0.0006
NUM_EPOCHS = 12
MAX_LENGTH = 25
DROPOUT = 0.25

def prepare_data(csv_file, ratio=0.1):
    with open(csv_file, 'r', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
        sampled_rows = random.sample(rows, int(len(rows) * ratio))
        data = [(r['mr'], r['ref']) for r in sampled_rows]
    
    word_vocab = {'<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3}
    all_words = []
    for _, ref in data:
        words = re.findall(r'\w+|[^\w\s]', ref.lower())
        all_words.extend(words)
    
    word_counts = Counter(all_words)
    for word, count in word_counts.most_common(2500):
        if count >= 2:
            word_vocab[word] = len(word_vocab)
    
    return data, word_vocab

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, EMBED_SIZE)
        self.pos_embedding = nn.Embedding(100, EMBED_SIZE)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_SIZE, nhead=NUM_HEADS, dim_feedforward=512, dropout=DROPOUT, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, NUM_LAYERS)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=EMBED_SIZE, nhead=NUM_HEADS, dim_feedforward=512, dropout=DROPOUT, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, NUM_LAYERS)
        
        self.fc = nn.Linear(EMBED_SIZE, vocab_size)
    
    def forward(self, src, tgt=None):
        src_embed = self.embedding(src) + self.pos_embedding(torch.arange(src.size(1), device=device))
        memory = self.encoder(src_embed)
        
        if tgt is not None:
            tgt_embed = self.embedding(tgt) + self.pos_embedding(torch.arange(tgt.size(1), device=device))
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.size(1)).to(device)
            output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
            return self.fc(output)
        else:
            batch_size = src.size(0)
            output_tokens = torch.full((batch_size, MAX_LENGTH), 0, device=device)
            output_tokens[:, 0] = 1
            
            for i in range(1, MAX_LENGTH):
                tgt = output_tokens[:, :i]
                tgt_embed = self.embedding(tgt) + self.pos_embedding(torch.arange(i, device=device))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(i).to(device)
                output = self.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                next_token = self.fc(output[:, -1:]).argmax(-1)
                output_tokens[:, i] = next_token.squeeze()
            
            return output_tokens

def train_model():
    print("Training...")
    data, word_vocab = prepare_data('trainset.csv', 0.1)
    
    model = SimpleTransformer(len(word_vocab)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0
        random.shuffle(data)
        
        for mr, ref in data:
            mr_words = re.findall(r'\w+', mr.lower())
            src = torch.tensor([word_vocab.get(w, word_vocab['<UNK>']) for w in mr_words], device=device).unsqueeze(0)
            
            ref_words = re.findall(r'\w+|[^\w\s]', ref.lower())
            tgt = torch.tensor([word_vocab['<SOS>']] + [word_vocab.get(w, word_vocab['<UNK>']) for w in ref_words], device=device).unsqueeze(0)
            tgt_out = torch.tensor([word_vocab.get(w, word_vocab['<UNK>']) for w in ref_words] + [word_vocab['<EOS>']], device=device).unsqueeze(0)
            
            optimizer.zero_grad()
            output = model(src, tgt)
            loss = criterion(output.view(-1, output.size(-1)), tgt_out.view(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data):.4f}")
    
    torch.save({'model_state': model.state_dict(), 'word_vocab': word_vocab}, 'neuralsys.pt')
    return model, word_vocab

def generate_text(model, mr, word_vocab, beam_width=3):
    model.eval()
    with torch.no_grad():
        # Prepare source
        mr_words = re.findall(r'\w+', mr.lower())
        src = torch.tensor([word_vocab.get(w, word_vocab['<UNK>']) for w in mr_words], device=device).unsqueeze(0)
        
        # Encode source
        src_embed = model.embedding(src) + model.pos_embedding(torch.arange(src.size(1), device=device))
        memory = model.encoder(src_embed)
        
        # Beam search initialization(i used as alt. to grid)
        beams = [([word_vocab['<SOS>']], 0.0)]  # (sequence, score)
        completed = []
        
        for step in range(MAX_LENGTH):
            new_beams = []
            
            for seq, score in beams:
                if seq[-1] == word_vocab['<EOS>']:
                    completed.append((seq, score))
                    continue
                
                # Prepare decoder input
                tgt = torch.tensor(seq, device=device).unsqueeze(0)
                tgt_embed = model.embedding(tgt) + model.pos_embedding(torch.arange(len(seq), device=device))
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(len(seq)).to(device)
                
                # Decode
                output = model.decoder(tgt_embed, memory, tgt_mask=tgt_mask)
                logits = model.fc(output[:, -1:])
                probs = torch.softmax(logits, dim=-1)
                
                # Get top 
                top_probs, top_indices = probs.topk(beam_width, dim=-1)
                
                for i in range(beam_width):
                    new_seq = seq + [top_indices[0, 0, i].item()]
                    new_score = score + torch.log(top_probs[0, 0, i]).item()
                    new_beams.append((new_seq, new_score))
            
            # Keep best beams
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        # Get the best sequence
        all_candidates = completed + beams
        if not all_candidates:
            return "A restaurant."
        best_seq = max(all_candidates, key=lambda x: x[1])[0]
        
        # Convert to text
        idx_to_word = {v: k for k, v in word_vocab.items()}
        words = []
        for idx in best_seq[1:]:  # Skip SOS
            if idx == word_vocab['<EOS>']:
                break
            words.append(idx_to_word.get(idx, '<UNK>'))
        
        return ' '.join(words) if words else "A restaurant."

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python neuralsys.py testset_w_refs.csv")
        sys.exit(1)
    
    try:
        checkpoint = torch.load('neuralsys.pt', map_location=device)
        word_vocab = checkpoint['word_vocab']
        model = SimpleTransformer(len(word_vocab)).to(device)
        model.load_state_dict(checkpoint['model_state'])
        print("Loaded pre-trained model")
    except:
        model, word_vocab = train_model()
    
    references, hypotheses = [], []
    with open(sys.argv[1], 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            generated = generate_text(model, row['mr'], word_vocab, beam_width=3)
            hypotheses.append(generated.split())
            references.append([row['ref'].split()])
            #print sentences to visulaize performance( this is how i checked to see how my model was performing)
            if i < 10:
                print(f"MR: {row['mr']}")
                print(f"Generated: {generated}")
                print(f"Reference: {row['ref']}")
                print("---")
    
    bleu_score = corpus_bleu(references, hypotheses, smoothing_function=SmoothingFunction().method1)
    print(f"BLEU score: {bleu_score:.4f}")