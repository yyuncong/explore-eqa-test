import os
import json

'''
To keep
with "exercise"

To discard
" wall"
'''

# train_dir = "/project/pi_chuangg_umass_edu/yuncong/data_v2/scene_datasets/hm3d/train"
# val_dir = "/work/pi_chuangg_umass_edu/yuncong/data/scene_datasets/hm3d/val"

# all_scenes = os.listdir(train_dir) + os.listdir(val_dir)
# all_scenes = [scene for scene in all_scenes if "-" in scene]

# all_classes = []
# total_scene_count = 0
# for scene_id in all_scenes:
#     if int(scene_id.split('-')[0]) < 800:
#         scene_path = os.path.join(train_dir, scene_id)
#     else:
#         scene_path = os.path.join(val_dir, scene_id)

#     scene_name = scene_id.split('-')[1]
#     annot_path = os.path.join(scene_path, scene_name + '.semantic.txt')

#     if not os.path.exists(annot_path):
#         continue

#     print(f"Reading {annot_path}")
#     total_scene_count += 1
#     with open(annot_path, 'r') as f:
#         all_lines = f.readlines()[1:]
#         classes = [line.split(",")[2].replace("\"", "") for line in all_lines]
#         classes = list(set(classes))
#         classes = [cls for cls in classes if cls != "unknown"]
#         print(f"Scene {scene_id} has classes {len(classes)}")
#         all_classes.extend(classes)

# all_classes = list(set(all_classes))
# print(f"Total classes: {len(all_classes)}")
# class_id_to_class_name = {i: cls for i, cls in enumerate(all_classes)}

# class_name_to_class_id = {cls: i for i, cls in enumerate(all_classes)}
# class_id_to_obj_count = {i: 0 for i in range(len(all_classes))}
# for scene_id in all_scenes:
#     if int(scene_id.split('-')[0]) < 800:
#         scene_path = os.path.join(train_dir, scene_id)
#     else:
#         scene_path = os.path.join(val_dir, scene_id)

#     scene_name = scene_id.split('-')[1]
#     annot_path = os.path.join(scene_path, scene_name + '.semantic.txt')

#     if not os.path.exists(annot_path):
#         continue

#     print(f"Reading {annot_path}")
#     with open(annot_path, 'r') as f:
#         all_lines = f.readlines()[1:]
#         classes = [line.split(",")[2].replace("\"", "") for line in all_lines]
#         for class_name in classes:
#             if class_name == "unknown":
#                 continue
#             class_id_to_obj_count[class_name_to_class_id[class_name]] += 1

# stat_res = {class_id: {"class_name": class_id_to_class_name[class_id], "count": class_id_to_obj_count[class_id]} for class_id in class_id_to_obj_count}
# json.dump(stat_res, open("class_stat.json", "w"), indent=4)
# print(f"Total scene count: {total_scene_count}")


'''
class_ignore = [
    "wall /outside", 
    " floor"
    end with " wall"
    "rug"
    " ceiling"
    "ceiling molding"
    "door frame"
    with "carpet"
    end with " frame"
]


must_get_bbox = [
    "mirror",
    "heater"
    " table"
    "with chair"    
    "fireplace"
    "tv"
    end with " lamp"


    "refrigerator",
    "microwave",
    "oven",

    "bed"
]

'''


def filter_class_name(class_names):
    filtered_class_names = []
    for name in class_names:
        if name in ['wall /outside', 'rug', 'ceiling molding']:
            continue
        if name.endswith(' floor') or name.endswith(' wall'):
            continue
        if ' ceiling' in name or 'carpet' in name or 'frame' in name:
            continue
        filtered_class_names.append(name)
    return filtered_class_names


def main():
    bbox_dir = '/home/yuncongyang_umass_edu/scene_understanding/hm3d_obj_bbox_all'
    all_files = os.listdir(bbox_dir)

    class_name_to_obj_count = {}

    bbox_count = 0
    for file_name in all_files:
        bbox_count += 1
        print(f"Processing {bbox_count}/{len(all_files)}, {file_name}")
        bbox_data = json.load(open(os.path.join(bbox_dir, file_name), "r"))
        for obj in bbox_data:
            class_name = obj["class_name"]
            if class_name == "unknown":
                continue
            if class_name not in class_name_to_obj_count:
                class_name_to_obj_count[class_name] = 0
            class_name_to_obj_count[class_name] += 1

    # filter out the classes that have less than 5 detections
    class_name_to_obj_count = {cls: class_name_to_obj_count[cls] for cls in class_name_to_obj_count if
                               class_name_to_obj_count[cls] >= 5}
    filtered_class_names = filter_class_name(list(class_name_to_obj_count.keys()))
    class_name_to_obj_count = {k: v for k, v in class_name_to_obj_count.items() if k in filtered_class_names}

    all_classes = list(class_name_to_obj_count.keys())
    class_id_to_class_name = {i: cls for i, cls in enumerate(all_classes)}

    stat_res = {class_id: {"class_name": class_id_to_class_name[class_id],
                           "count": class_name_to_obj_count[class_id_to_class_name[class_id]]} for class_id in
                class_id_to_class_name}
    # sort the stat_res by count
    stat_res = {k: v for k, v in sorted(stat_res.items(), key=lambda item: item[1]["count"], reverse=True)}

    json.dump(stat_res, open("yolo_finetune/class_stat.json", "w"), indent=4)
    json.dump(class_id_to_class_name, open("yolo_finetune/class_id_to_class_name.json", "w"), indent=4)

    print(f"num class with detection < 5: {len([cls for cls in class_name_to_obj_count if class_name_to_obj_count[cls] < 5])}")


if __name__ == "__main__":
    main()