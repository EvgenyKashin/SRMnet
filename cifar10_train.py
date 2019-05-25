import argparse
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from back import Bone, utils
from datasets import cifar10
from models.resnet import resnet20
from models.custom_resnet import se_resnet20, srm_resnet20

data_dir = 'cifar10'
model_names = ['resnet', 'senet', 'srmnet']
num_classes = 10
batch_size = 128
epochs_count = 240
num_workers = 12

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', required=True, choices=model_names)
args = parser.parse_args()

datasets = cifar10.get_datasets(data_dir)

if args.model_name == 'resnet':
    model = resnet20(num_classes=num_classes)  # 0.877 - 272,474
elif args.model_name == 'senet':
    model = se_resnet20(num_classes=num_classes)  # 0.877 - 274,490(+0.7%)
elif args.model_name == 'srmnet':
    model = srm_resnet20(num_classes=num_classes)  # and 0.871 - 273,818( +0.5%)

optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                      weight_decay=1e-4)

scheduler = StepLR(optimizer, 40, 0.2)
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
