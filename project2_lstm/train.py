import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.profiler import profile, record_function, ProfilerActivity
import time
import argparse
import json
from pathlib import Path
import numpy as np
from collections import Counter
import re


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, num_classes=2, dropout=0.5):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        out = self.dropout(hidden)
        out = self.fc(out)
        return out


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_length=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = self.texts[idx].lower()
        tokens = re.findall(r'\b\w+\b', text)
        indices = [self.vocab.get(token, 1) for token in tokens[:self.max_length]]
        
        if len(indices) < self.max_length:
            indices += [0] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class LSTMTrainer:
    def __init__(self, device, batch_size=64, max_length=200, vocab_size=10000):
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.vocab = None
        self.setup_data()
        self.setup_model()
    
    def build_vocab(self, texts):
        all_tokens = []
        for text in texts:
            tokens = re.findall(r'\b\w+\b', text.lower())
            all_tokens.extend(tokens)
        
        counter = Counter(all_tokens)
        most_common = counter.most_common(self.vocab_size - 2)
        
        vocab = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, _) in enumerate(most_common, start=2):
            vocab[word] = idx
        
        return vocab
    
    def setup_data(self):
        print("Loading synthetic sentiment data...")
        
        positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'best', 'perfect']
        negative_words = ['bad', 'terrible', 'awful', 'horrible', 'worst', 'hate', 'disappointing', 'poor']
        neutral_words = ['movie', 'film', 'story', 'actor', 'scene', 'plot', 'character', 'ending']
        
        def generate_text(sentiment, length=50):
            words = neutral_words.copy()
            if sentiment == 1:
                words.extend(positive_words * 3)
            else:
                words.extend(negative_words * 3)
            return ' '.join(np.random.choice(words, length))
        
        n_train, n_test = 2000, 400
        
        train_texts = [generate_text(i % 2) for i in range(n_train)]
        train_labels = [i % 2 for i in range(n_train)]
        
        test_texts = [generate_text(i % 2) for i in range(n_test)]
        test_labels = [i % 2 for i in range(n_test)]
        
        print(f"Building vocabulary from {len(train_texts)} samples...")
        self.vocab = self.build_vocab(train_texts)
        print(f"Vocabulary size: {len(self.vocab)}")
        
        train_dataset = TextDataset(train_texts, train_labels, self.vocab, self.max_length)
        test_dataset = TextDataset(test_texts, test_labels, self.vocab, self.max_length)
        
        self.trainloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
            num_workers=0, pin_memory=False
        )
        self.testloader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
            num_workers=0, pin_memory=False
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Test samples: {len(test_dataset)}")
    
    def setup_model(self):
        print("Initializing LSTM model...")
        self.model = LSTMClassifier(
            vocab_size=len(self.vocab),
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            num_classes=2
        )
        self.model = self.model.to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Total parameters: {total_params:,}")
    
    def train_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(self.trainloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch: {epoch} [{batch_idx}/{len(self.trainloader)}] '
                      f'Loss: {running_loss/(batch_idx+1):.3f} '
                      f'Acc: {100.*correct/total:.2f}%')
        
        return running_loss / len(self.trainloader), 100. * correct / total
    
    def evaluate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        return test_loss / len(self.testloader), 100. * correct / total
    
    def train_with_profiling(self, epochs, profile_enabled=True):
        results = {
            'device': self.device,
            'epochs': epochs,
            'batch_size': self.batch_size,
            'max_length': self.max_length,
            'vocab_size': len(self.vocab),
            'train_losses': [],
            'train_accs': [],
            'test_losses': [],
            'test_accs': [],
            'epoch_times': [],
            'total_time': 0,
            'profiler_path': None
        }
        
        start_time = time.time()
        
        if profile_enabled and self.device == 'cuda':
            torch.cuda.synchronize()
        
        profiler_dir = Path(f'runs/lstm_{self.device}')
        profiler_dir.mkdir(parents=True, exist_ok=True)
        
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA] if self.device == 'cuda' else [ProfilerActivity.CPU],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir))
        ) as prof:
            for epoch in range(epochs):
                epoch_start = time.time()
                
                with record_function("train_epoch"):
                    train_loss, train_acc = self.train_epoch(epoch)
                
                with record_function("evaluate"):
                    test_loss, test_acc = self.evaluate()
                
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                epoch_time = time.time() - epoch_start
                
                results['train_losses'].append(train_loss)
                results['train_accs'].append(train_acc)
                results['test_losses'].append(test_loss)
                results['test_accs'].append(test_acc)
                results['epoch_times'].append(epoch_time)
                
                print(f'\nEpoch {epoch}: Train Loss: {train_loss:.3f}, Train Acc: {train_acc:.2f}%, '
                      f'Test Loss: {test_loss:.3f}, Test Acc: {test_acc:.2f}%, Time: {epoch_time:.2f}s')
                
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                prof.step()
        
        if self.device == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        results['total_time'] = total_time
        results['profiler_path'] = str(profiler_dir)
        
        print(f'\nTotal training time: {total_time:.2f}s')
        print(f'Average time per epoch: {total_time/epochs:.2f}s')
        
        results_path = profiler_dir / 'results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f'Results saved to {results_path}')
        
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'])
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--max-length', type=int, default=200)
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    print(f"Training LSTM on {args.device.upper()}")
    print(f"Batch size: {args.batch_size}, Epochs: {args.epochs}, Max length: {args.max_length}")
    
    trainer = LSTMTrainer(args.device, args.batch_size, args.max_length)
    results = trainer.train_with_profiling(args.epochs)
    
    print(f"\nTraining completed on {args.device.upper()}")
    print(f"Final test accuracy: {results['test_accs'][-1]:.2f}%")
    
    if args.device == 'cuda':
        torch.cuda.empty_cache()
        print("GPU memory cleared")


if __name__ == '__main__':
    main()
