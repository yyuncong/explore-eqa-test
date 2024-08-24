import os
import random

data_dir = '/gpfs/u/home/LMCG/LMCGnngn/scratch/yanghan/3d/explore-eqa-test/yolo_finetune_data'
train_ratio = 0.9

all_images = os.listdir(os.path.join(data_dir, 'images'))

train_images = random.sample(all_images, int(len(all_images) * train_ratio))
val_images = list(set(all_images) - set(train_images))

print(f"Total images: {len(all_images)}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

with open(os.path.join(data_dir, 'train.txt'), 'w') as f:
    for image in train_images:
        f.write(f"./image/{image}\n")

with open(os.path.join(data_dir, 'val.txt'), 'w') as f:
    for image in val_images:
        f.write(f"./image/{image}\n")