from torchvision import datasets
import torchvision.transforms as tf
from torch.utils.data import DataLoader,Dataset
import torch


class CachedDatset(Dataset):
    def __init__(self,dataset,data_frac=1):
        self.data=torch.stack([x for x,y in dataset])
        self.labels=torch.tensor([y for x,y in dataset])
        self.shuffled_idx=torch.randperm(len(self.labels))
        self.data_frac=data_frac
        print(f"Cached — {self.data.nbytes / 1024**2:.1f} MB in RAM")


    
    def __len__(self):
        return int(len(self.labels)*self.data_frac)
    
    def __getitem__(self,idx):
        return self.data[self.shuffled_idx[idx]],self.labels[self.shuffled_idx[idx]]


def load_data(data_frac=1):
    train_transform=tf.Compose([
                    tf.ToTensor(),
                    ])

    test_transform=tf.Compose([
                    tf.ToTensor(),tf.Normalize((0.4914, 0.4822, 0.4465),
                 (0.2470, 0.2435, 0.2616))
                    ])


    train_data = datasets.CIFAR10(
        root="../cifar10_dataset",
        train=True,
        download=True,
        transform=train_transform,
    )

    test_data = datasets.CIFAR10(
        root="../cifar10_dataset",
        train=False,
        download=True,
        transform=test_transform
    )
    

    return CachedDatset(train_data,data_frac),CachedDatset(test_data)


def load_dataloaders(batch_size=64,shuffle=True,num_workers=8,pin_memory=True,prefetch_factor=2,persistent_workers=True,data_frac=1):
    train_data,test_data=load_data(data_frac)
    train_loader=DataLoader(train_data,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor,
                            persistent_workers=persistent_workers)
    
    test_loader=DataLoader(test_data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=num_workers,
                        pin_memory=pin_memory,
                        prefetch_factor=prefetch_factor,
                        persistent_workers=persistent_workers)

    return train_loader,test_loader



if __name__=="__main__":
    train_loader,test_loader=load_dataloaders(256,data_frac=0.1)
    print(len(train_loader))


