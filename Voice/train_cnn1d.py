import os
import argparse
from dataset import VoiceCommandDataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from models import MLPClassifier, CNN1DSoundClassifier
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import random
import torchaudio
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()



def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')

def collate_fn_pad(batch):
    waveforms, labels = zip(*batch)
    max_len = max([w.shape[-1] for w in waveforms])
    padded = [F.pad(w, (0, max_len - w.shape[-1])) for w in waveforms]
    padded = torch.stack(padded)
    labels = torch.tensor(labels, dtype=torch.long)  # integer label for CrossEntropy
    return padded, labels

def train(args, io):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = "data/english"
    train_samples, val_samples, test_samples = split_dataset(root_dir)

    classes = sorted(os.listdir(root_dir))
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
    num_classes = len(classes)

    # Optional transform
    transform = torchaudio.transforms.MFCC(
        sample_rate=16000,
        n_mfcc=40,                  # output dimension
        melkwargs={
            "n_mels": 40,           # was 128
            "n_fft": 400,           # default for 25ms @16kHz
            "hop_length": 160,      # default for 10ms hop
        }
    )

    # Create datasets
    train_dataset = VoiceCommandDataset(train_samples, class_to_idx, transform)

    val_dataset = VoiceCommandDataset(val_samples, class_to_idx, transform)
    test_dataset = VoiceCommandDataset(test_samples, class_to_idx, transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn_pad)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn_pad)

    input_size = 40  # from n_mfcc=40
    model = CNN1DSoundClassifier(input_channels=input_size, num_classes=num_classes).to(device)
    # model = MLPClassifier(input_size=input_size, num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = args.epochs

    train_losses = []
    val_losses = []
    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for waveforms, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            waveforms, labels = waveforms.to(device), labels.to(device)
            waveforms = waveforms.squeeze(1) if waveforms.ndim == 3 else waveforms
            features = waveforms.view(waveforms.size(0), waveforms.size(2), waveforms.size(3))  # [B, 1, 40, T] → [B, 40, T]
            
            preds = model(features)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        io.cprint(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

        train_losses.append(total_loss)

        # Validation accuracy
        model.eval()
        all_preds, all_labels = [], []
        val_loss_total = 0.0
        with torch.no_grad():
            for waveforms, labels in val_loader:
                waveforms, labels = waveforms.to(device), labels.to(device)
                waveforms = waveforms.squeeze(1) if waveforms.ndim == 3 else waveforms
                features = waveforms.view(waveforms.size(0), waveforms.size(2), waveforms.size(3))  # [B, 1, 40, T] → [B, 40, T]
                
                preds = model(features)

                val_loss = criterion(preds, labels)
                val_loss_total += val_loss.item()
            
                pred_classes = preds.argmax(dim=1)



                all_preds.extend(pred_classes.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())  #

        val_losses.append(val_loss_total)
        test_acc = accuracy_score(all_labels, all_preds)

        if test_acc >= best_acc:
            best_acc = test_acc
            print("Best Test Acc: ",best_acc)
            torch.save(model.state_dict(), 'checkpoints/%s/models/best_model.pth' % args.exp_name)
        torch.save(model.state_dict(), 'checkpoints/%s/models/model_final.pth' % args.exp_name)

        io.cprint(f"Validation Accuracy: {test_acc:.4f} - Val Loss: {val_loss_total:.4f}")

    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'checkpoints/{args.exp_name}/loss_curve.png')
    plt.show()
def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--resize', type=int, default=112, help='Resized image')
    parser.add_argument('--frames_per_clip', type=int, default=16, help='frames_per_clip')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--start_epoch', type=int, default=0, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', action="store_true", help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--gpu', type=int, default=0,
                        help='which CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool, default=False,
                        help='evaluate the model')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight Decay')
    parser.add_argument('--resume', action="store_true", help='resume from checkpoint')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    return parser.parse_args()


def split_dataset(root_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    assert train_ratio + val_ratio + test_ratio == 1.0, "Split ratios must sum to 1"
    random.seed(seed)

    classes = sorted(os.listdir(root_dir))
    dataset = []

    for cls in classes:
        cls_path = os.path.join(root_dir, cls)
        if not os.path.isdir(cls_path):
            continue
        files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.wav')]
        for f in files:
            dataset.append((f, cls))

    # Split
    train_val, test = train_test_split(dataset, test_size=test_ratio, random_state=seed, stratify=[x[1] for x in dataset])
    train, val = train_test_split(train_val, test_size=val_ratio / (train_ratio + val_ratio), random_state=seed, stratify=[x[1] for x in train_val])

    return train, val, test




if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    train(args, io)
    