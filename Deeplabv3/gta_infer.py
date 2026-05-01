import torch
import torch.nn as nn
from torch.utils import data
from torchvision import transforms as T
import os
import argparse
import network 
from metrics import StreamSegMetrics 
from tqdm import tqdm

from gta import GTA

'''
python Deeplabv3/gta_infer.py \
  --data_root ~/inAI_plusStyle/data/gta \
  --ckpt ~/inAI_plusStyle/Deeplabv3/checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth
'''
def get_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="path to your pair folder")
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet', help='model name')
    parser.add_argument("--ckpt", type=str, default='best_deeplabv3plus_mobilenet_cityscapes_os16.pth', required=True, help="path to cityscapes trained checkpoint")
    parser.add_argument("--num_classes", type=int, default=19, help="Cityscapes classes")
    parser.add_argument("--output_stride", type=int, default=16)
    parser.add_argument("--gpu_id", type=str, default='0')
    parser.add_argument("--batch_size", type=int, default=10)

    return parser

def main():
    opts = get_argparser().parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # 1. Define Transforms (Normalization as used in training)
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 2. Load Dataset
    
    test_dst = GTA(root=opts.data_root, transform=transform)
    test_loader = data.DataLoader(test_dst, batch_size=opts.batch_size, shuffle=False, num_workers=0)

    print("Dataset size:", len(test_dst))

    # 3. Load Model
    model_map = {
        'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
    }
    
    model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    
    # Load Checkpoint
    checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'), weights_only=False)
    # Handle DataParallel wrapping if necessary
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
    else:
        state_dict = checkpoint
        
    # Remove 'module.' prefix if it exists
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model) # 2. 모델을 DataParallel로 감싸기
    
    model.to(device)
    model.eval()

    # 4. Run Evaluation
    metrics = StreamSegMetrics(opts.num_classes)
    metrics.reset()

    print("Start validation...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)

            outputs = model(images)
            preds = outputs.detach().max(dim=1)[1].cpu().numpy()
            targets = labels.cpu().numpy()

            metrics.update(targets, preds)

    score = metrics.get_results()
    print("-------------------------")
    # print(f"Overall Acc: {score['Overall Acc']}") # 
    # print("Class IoU:", score['Class IoU'])
    print(f"Mean IoU: {score['Mean IoU']}")
    print("-------------------------")

if __name__ == '__main__':
    main()