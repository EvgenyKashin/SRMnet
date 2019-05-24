import torchvision
import torchvision.transforms as transforms

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


def get_datasets(data_dir):
    return {
        'train': torchvision.datasets.CIFAR10(root=data_dir,
                                              train=True,
                                              download=True,
                                              transform=transform_train),
        'val': torchvision.datasets.CIFAR10(root=data_dir,
                                            train=False,
                                            download=True,
                                            transform=transform_test)
    }