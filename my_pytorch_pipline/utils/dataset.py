import torchvision
from torch.utils.data import Dataset, DataLoader

def load_Minst(dataset_path = "/home/hpczeji1/hpc-work/Codebase/Datasets/mnist_data"):
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5), std = (0.5))  # normalize to [-1,1]
    ])

    from torchvision.datasets import MNIST
    train_dataset = MNIST(root=dataset_path,
                      train=True,
                      transform=transform,
                      target_transform=None,  # Eg1.2.1 : <class 'int'>
                      download=False)
    
    test_dataset = MNIST(root=dataset_path,
                      train=False,
                      transform=transform,
                      target_transform=None,  # Eg1.2.1 : <class 'int'>
                      download=False)

    return train_dataset, test_dataset
    