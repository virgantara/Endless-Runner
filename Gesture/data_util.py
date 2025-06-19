from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split

def split_dataset(dataset, test_size=0.2):
    targets = [s[1] for s in dataset.samples]
    train_idx, test_idx = train_test_split(
        list(range(len(dataset))),
        test_size=test_size,
        stratify=targets,
        random_state=42
    )
    return Subset(dataset, train_idx), Subset(dataset, test_idx)