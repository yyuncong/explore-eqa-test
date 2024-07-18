import os
import glob
import json
from tqdm import tqdm


folder = 'exploration_data'
json_files = glob.glob(os.path.join(folder, "*", "00*.json"))

for json_path in tqdm(json_files):
    with open(json_path, 'r') as f:
        data = json.load(f)
    for snapshot_data in data['snapshots']:
        if 0 in snapshot_data['obj_ids']:
            print(json_path)
