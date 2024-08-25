import os
import random

data_dir = '/gpfs/u/home/LMCG/LMCGhazh/scratch/yanghan/explore-eqa-test/yolo_finetune_data'
train_ratio = 0.9

all_images = os.listdir(os.path.join(data_dir, 'images'))


# random split
# train_images = random.sample(all_images, int(len(all_images) * train_ratio))
# val_images = list(set(all_images) - set(train_images))

# split by scene id
train_images = [name for name in all_images if int(name.split('--')[1].split('-')[0]) < 800]
val_images = [name for name in all_images if int(name.split('--')[1].split('-')[0]) >= 800]


print(f"Total images: {len(all_images)}")
print(f"Train images: {len(train_images)}")
print(f"Validation images: {len(val_images)}")

with open(os.path.join(data_dir, 'train_x.txt'), 'w') as f:
    for image in train_images:
        f.write(f"./images/{image}\n")

with open(os.path.join(data_dir, 'val_x.txt'), 'w') as f:
    for image in val_images:
        f.write(f"./images/{image}\n")