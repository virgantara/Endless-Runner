import os
import argparse
from dataset import VideoDatasetCV
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from models import CNN_LSTM
from tqdm import tqdm
import torch.nn.functional as F

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


def train(args, io):
    data_dir = 'data/Full-body_Gestures'
    dataset = VideoDatasetCV(data_dir)
    train_set, test_set = split_dataset(dataset)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)  

    num_classes = len(dataset.class_to_idx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_LSTM(num_classes=num_classes)

    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0


    for epoch in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = F.cross_entropy(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()

            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            loss_sum += loss.item() * y.size(0)

        train_acc = correct / total
        print(f"[Epoch {epoch+1}] Train Loss: {loss_sum / total:.4f}, Train Acc: {train_acc:.4f}")


        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            correct, total = 0, 0
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                pred = out.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        test_acc = correct / total
        print(f"[Epoch {epoch+1}] Test Acc: {test_acc:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved at epoch {epoch+1} with test acc {best_acc:.4f}")

def parse_args():
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--exp_name', type=str, default='exp', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--model', type=str, default='dgcnn_cls', metavar='N',
                        choices=['dgcnn_cls', 'dgcnn_seg', 'pointnet_cls', 'pointnet_seg'],
                        help='Model to use, [pointnet, dgcnn]')
    parser.add_argument('--dataset_name', type=str, default='modelnet40svm', metavar='N',
                        choices=['modelnet40svm', 'scanobjectnnsvm'],
                        help='Dataset name to test, [modelnet40svm, scanobjectnnsvm]')
    parser.add_argument('--batch_size', type=int, default=4, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
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

def split_dataset(dataset, test_size=0.2):
    targets = [s[1] for s in dataset.samples]
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=targets,
        random_state=42
    )
    return Subset(dataset, train_idx), Subset(dataset, test_idx)


if __name__ == "__main__":
    args = parse_args()
    _init_()

    device = torch.device(f"cuda:{args.gpu}")

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    torch.manual_seed(args.seed)

    train(args, io)
    