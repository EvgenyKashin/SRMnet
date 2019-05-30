import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from back import Bone, utils
from datasets import imagenet
from models.resnet_with_block import resnet50, se_resnet50, srm_resnet50

data_dir = 'imagenet'
model_names = ['resnet', 'senet', 'srmnet']
num_classes = 1000
batch_size = 32
epochs_count = 100
num_workers = 8

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True, choices=model_names)
args = parser.parse_args()

datasets = imagenet.get_datasets(data_dir)

if args.model_name == 'resnet':
    model = resnet50(num_classes=num_classes)
elif args.model_name == 'senet':
    model = se_resnet50(num_classes=num_classes)
elif args.model_name == 'srmnet':
    model = srm_resnet50(num_classes=num_classes)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4)

scheduler = StepLR(optimizer, 30, 0.1)
criterion = nn.CrossEntropyLoss()

backbone = Bone(model,
                datasets,
                criterion,
                optimizer,
                scheduler=scheduler,
                scheduler_after_ep=False,
                metric_fn=utils.accuracy_metric,
                metric_increase=True,
                batch_size=batch_size,
                num_workers=num_workers,
                weights_path=f'weights/imagenet_best_{args.model_name}.pth',
                log_dir=f'logs/imagenet/{args.model_name}')

backbone.fit(epochs_count)
