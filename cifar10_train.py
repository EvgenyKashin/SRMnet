import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from back import Bone, utils
from datasets import cifar10
from models.resnet_with_block import resnet32, se_resnet32, srm_resnet32

data_dir = 'cifar10'
model_names = ['resnet', 'senet', 'srmnet']
num_classes = 10
batch_size = 128
epochs_count = 100
num_workers = 8

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True, choices=model_names)
args = parser.parse_args()

datasets = cifar10.get_datasets(data_dir)

if args.model_name == 'resnet':
    model = resnet32(num_classes=num_classes)
elif args.model_name == 'senet':
    model = se_resnet32(num_classes=num_classes)
elif args.model_name == 'srmnet':
    model = srm_resnet32(num_classes=num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4)

scheduler = MultiStepLR(optimizer, [70, 80], 0.1)
criterion = nn.CrossEntropyLoss()

backbone = Bone(model,
                datasets,
                criterion,
                optimizer,
                scheduler=scheduler,
                scheduler_after_ep=False,
                # early_stop_epoch=40,
                metric_fn=utils.accuracy_metric,
                metric_increase=True,
                batch_size=batch_size,
                num_workers=num_workers,
                weights_path=f'weights/best_{args.model_name}.pth',
                log_dir=f'logs/{args.model_name}')

backbone.fit(epochs_count)
