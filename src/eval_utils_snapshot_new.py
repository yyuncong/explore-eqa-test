from torch.utils.data.distributed import DistributedSampler
import os
import json
import torch
from PIL import Image
from torchvision import transforms
from collections import defaultdict
from easydict import EasyDict
from transformers import AutoTokenizer
from torch.utils.data import DataLoader,Dataset,Subset
from itertools import chain
import random
import numpy as np
import math

# Need to reorganize this file

SCENE_TOKEN = "<scene>"
# FRONTIER_TOKEN = "<frontier>"
SELECT_TOKEN = "<select>"
SCENE_TOKEN = "<scene>"
VISUAL_TOKEN = "<visual>"
TACTILE_TOKEN = "<temperature>"
SOUND_TOKEN = "<sound>"
# TEMP_TOKEN = "<temperature>"
GET_VISUAL_TOKEN = "<observe>"
GET_TACTILE_TOKEN = "<touch>"
GET_SOUND_TOKEN = "<tap>"
SELECT_TOKEN = "<select>"

# because sos token is added, the max_length should be +1?

NUM_BINS = 128
COORD_RANGE = (-7, 7)

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

PE = positionalencoding2d(1024, NUM_BINS, NUM_BINS)

def merge_patches(patches, patch_size):
    num_patches, num_patches, patch_dim = patches.shape
    new_num_patches = num_patches // patch_size
    assert num_patches % patch_size == 0
    patches = patches.view(
        new_num_patches,
        patch_size,
        new_num_patches,
        patch_size,
        patch_dim,
    )
    patches = patches.permute(0, 2, 1, 3, 4).reshape(
        new_num_patches, new_num_patches, patch_size ** 2, patch_dim
    ).mean(-2)
    patches = patches.view(new_num_patches * new_num_patches, patch_dim)
    return patches

def discretize_coordinates(coords, num_bins=128, coord_range=(-10, 10)):
    # Ensure coords is a torch tensor
    if not isinstance(coords, torch.Tensor):
        coords = torch.tensor(coords, dtype=torch.float32)

    # Extract min and max values from the coord_range
    min_val, max_val = coord_range

    # Normalize coordinates to range [0, 1]
    normalized_coords = (coords - min_val) / (max_val - min_val)

    # Scale normalized coordinates to range [0, num_bins - 1]
    scaled_coords = normalized_coords * (num_bins - 1)

    # Round to get discrete bin indices and clamp to ensure within range
    discretized_coords = torch.round(scaled_coords).long()
    discretized_coords = torch.clamp(discretized_coords, 0, num_bins - 1)

    return discretized_coords

def sum_positional_encodings(x, pos, pe, num_bins=128, coord_range=(-10, 10)):
    '''
    x: (num_points, d_model)
    pos: (num_points, 2)
    pe: (d_model, num_bins, num_bins)
    '''
    # Discretize the coordinates
    discretized_coords = discretize_coordinates(pos, num_bins=num_bins, coord_range=coord_range).unsqueeze(0)
    # Get the positional encodings for the coordinates
    x_pe = pe[:, discretized_coords[:, :, 0], discretized_coords[:, :, 2]].permute(1, 2, 0).squeeze(0)
    # Sum the positional encodings along the num_points dimension
    x += x_pe
    return x

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint, False)
    del checkpoint
    
def load_ds_checkpoint(model, checkpoint_path, exclude_frozen_parameters = False):
    '''
    the input checkpointpoint path should be like ckpt_dir/tag/{ckpt files}
    if lora is used, the lora config should be placed inside the ckpt_dir
    '''
    from peft import LoraConfig, get_peft_model
    import deepspeed
    from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
    checkpoint_path = checkpoint_path.split('/')
    saving_folder, tag = '/'.join(checkpoint_path[:-1]), checkpoint_path[-1]
    state_dict = get_fp32_state_dict_from_zero_checkpoint(
        saving_folder,
        tag,
        exclude_frozen_parameters = exclude_frozen_parameters
    )
    if "lora_config.json" in os.listdir(saving_folder):
        with open(os.path.join(saving_folder, "lora_config.json"), 'r') as f:
            lora = json.load(f)
        # wrap up model with lora
        lora_config = LoraConfig(
            r = lora['r'],
            lora_alpha = lora['lora_alpha'],
            target_modules = lora['target_modules'],
            lora_dropout = lora['lora_dropout'],
            bias = lora['bias'],
            task_type = lora['task_type']
        )
        model = get_peft_model(model, lora_config)
    model.load_state_dict(state_dict, strict = False)
    del state_dict

def collate_wrapper(batch):
    max_length = max(b.length for b in batch) + 1
    max_scene_length = max(b.scene_feature.shape[0] for b in batch)
    # max_frontier_length = max(b.frontier_feature.shape[0] for b in batch)
    
    scene_feature = torch.zeros((len(batch), max_scene_length, 1024))
    scene_insert_loc = torch.zeros((len(batch), max_scene_length))
    
    for (j,b) in enumerate(batch):
        scene_feature[j, :b.scene_feature.shape[0]] = b.scene_feature
        # frontier_feature[j, :b.frontier_feature.shape[0]] = b.frontier_feature
        scene_insert_loc[j, :b.scene_insert_loc.shape[0]] = b.scene_insert_loc
    # print(batch[0].input_ids)
    return EasyDict(
        input_ids=torch.cat([b.input_ids for b in batch])[...,:max_length],
        attention_mask=torch.cat([b.attention_mask for b in batch])[...,:max_length],
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc.to(torch.long),
        scene_length = torch.tensor([b.scene_length for b in batch]),
        max_scene_length = torch.tensor([b.scene_feature.shape[0] for b in batch])
    )

def load_scene_features(scene_dir, scene_id):
    scene = {}
    scene_fold = os.path.join(scene_dir, scene_id)
    for object_f in os.listdir(scene_fold):
        try:
            object_id = object_f[:-3]
            object_feature  = torch.load(os.path.join(scene_fold, object_f),
                                        map_location = 'cpu')
            scene[object_id] = object_feature
        except:
            continue
    return scene

def rgba2rgb( rgba, background=(255,255,255) ):
    row, col, ch = rgba.shape

    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros( (row, col, 3), dtype='float32' )
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' ) / 255.0

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return np.asarray( rgb, dtype='uint8' )

def prepare_step_dict(step_dict):
    pass

def encode(model, image_processor, img):
    img = image_processor.preprocess(img, return_tensors='pt')['pixel_values']
    img = torch.cat([img], dim=0).half().cuda()
    img = model.encode_images(img)
    return img

# jiachen TODO: add prefiltering and parts from eval_dataset here
def prepare_prompt_before_snapshot(step):
    num_visual_tokens = step["num_visual_tokens"]
    text =  f"Question: {step['question']}\n" 
    multi_src_feature = []
    if step.get("use_egocentric_views") is True:
        text += "Followings are the egocentric views:\n "
        for i in range(len(step["egocentric_view_features"])):
            text += f"<scene>"
        text += " /\n"
        egocentric_features = step["egocentric_view_features"]
        if step.get("add_positional_encodings") is True:
            egocentric_positions = torch.cat(
                [
                    torch.tensor(step["position"] - step["position"])
                    for _ in range(egocentric_features.shape[0])
                ],
                dim=0
            )
            egocentric_features = sum_positional_encodings(
                egocentric_features, egocentric_positions, PE, num_bins=NUM_BINS, coord_range=COORD_RANGE
            )
        multi_src_feature.append(egocentric_features)
    
    # yuncong TODO: align this with the model later
    text += f"Select the frontier/snapshot that would help finding the answer of the question.\n"

    # TODO: add position to memory if possible?
    if step.get("use_action_memory") is True:
        text += f"Here is your selection in the previous step:\n "
        if step["memory_feature"] is None:
            text += f"No selection in the previous step. "
        else:
            text += f"<scene> "
        multi_src_feature.append(step["memory_feature"])
        text += "/\n"
        
    return text, multi_src_feature

def prepare_frontier(step):
    num_visual_tokens = step["num_visual_tokens"]
    text = "Below are all the frontiers that we can explore:\n"
    if len(step['frontiers']) > 0:
        for i, frontier in enumerate(step['frontiers']):
            text += f"frontier {i} "
            for _ in range(num_visual_tokens):
                text += f"<scene>"
            text += " / "
    else:
        text += f"No frontier available "
    text += "\n"
    frontier_features = step["frontier_features"]
    if len(step['frontiers']) > 0 and step.get("add_positional_encodings") is True:
        frontier_positions = torch.tensor(step["frontier_positions"])
        frontier_features = sum_positional_encodings(
            frontier_features, frontier_positions, PE, num_bins=NUM_BINS, coord_range=COORD_RANGE
        )
    
    return text, frontier_features
    
def prepare_snapshot_input(
    seen_classes,
    snapshot_classes,
    snapshot_features,
    prefiltering,
    ranking,
    topk,
    num_visual_tokens,
):
    import logging
    snapshot_index = len(snapshot_classes)
    # the mapping from transformed object index to original object index(used by tsdf)
    snap_indices = None

    # prefiltering TODO
    if prefiltering:
        ranking = [cls for cls in ranking if cls in seen_classes]
        ranking = ranking[:topk]
        ranking_set = set(ranking)
        # logging.info("filtered ranking")
        # logging.info('_'.join(ranking))
        snap_indices = [
            snap_idx
            for snap_idx in range(snapshot_index)
            if len(set(snapshot_classes[snap_idx]) & ranking_set) > 0
        ]
        #logging.info("snap_indices: "+' '.join([str(idx) for idx in snap_indices]))
        # logging.info("raw snapshot classes: "+'/'.join([','.join(sc) for sc in snapshot_classes]))
        snapshot_classes = [
            snapshot_classes[snap_idx] for snap_idx in snap_indices
        ]
        #print("snapshot classes before filtering", snapshot_classes)
        #snapshot_classes = [
        #    set(sc)&ranking_set for sc in snapshot_classes
        #]
        snapshot_classes = [
            [scls for scls in list(dict.fromkeys(snap_cls)) if scls in ranking_set] 
            for snap_cls in snapshot_classes
        ]
        #print("snapshot classes after filtering", snapshot_classes)
        snapshot_features = [
            snapshot_features[snap_idx] for snap_idx in snap_indices
        ]
        # logging.info("filtered snapshot classes: "+'/'.join([','.join(sc) for sc in snapshot_classes]))
        # Note that if apply prefiltering, we may have #(objects) < object_index
        # 4. reassign object_index = #(object)
        snapshot_index = len(snapshot_classes)

    text = "These are the snapshots:\n"
    for i, class_names in enumerate(snapshot_classes):
        text += f"snapshot {i} "
        class_names_set = set(class_names)
        class_names_list = list(class_names_set)
        sorted_class_names = sorted(class_names_list)
        for class_name in sorted_class_names:
            text += f"{class_name}, "
        for _ in range(num_visual_tokens):
            text += "<scene>"
        text += " / "

    if snapshot_index == 0:
        text += f"No snapshot available "
        # construct zero scene feature if all snapshots are missed
        snapshot_features = None
    else:
        snapshot_features = torch.cat(snapshot_features, dim=0)
    text += "\n"
    return text, snapshot_features, snapshot_index, snap_indices

def prepare_prefiltering_prompt(question, tokenizer, classes, max_length, topk):
    filter_text = f"Question: {question}\n"
    filter_text += "These are the objects available in current scene graph\n"
    for class_name in classes:
        filter_text += f"{class_name} \n"
    if len(classes) == 0:
        filter_text += "No object available \n"
    filter_text += f"Rank at most top {topk} of them from high to low based on their importance on answering the question\n"
    filter_text += "Answer: "
    # print("filtering prompt", len(filter_text))
    # print(filter_text)
    # Jiachen TODO 7: output filter_input_ids/filter_attention_mask/filter_length for the filtering question
    # print("raw text of filter prompt:", filter_text)
    filter_text = tokenizer(
        filter_text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    filter_input_ids = filter_text["input_ids"]
    filter_length = torch.nonzero(filter_input_ids).shape[0]
    filter_attention_mask = filter_text["attention_mask"]
    return filter_input_ids, filter_length, filter_attention_mask

# def test_tokenizer(tokenizer):
#     print("snapshot", tokenizer.encode("snapshot"))
#     print("object", tokenizer.encode("object"))
#     # print(tokenizer.decode(tokenizer.encode("bath tub")))
#     # print(tokenizer.encode("bath tub"))
#     # print(tokenizer.decode(tokenizer.encode(" bath tub")))
#     # print(tokenizer.encode(" bath tub"))
#     # print(tokenizer.decode(tokenizer.encode("<scene> bath tub")))
#     # print(tokenizer.encode("<scene> bath tub"))
#     # print(tokenizer.decode(tokenizer.encode("<scene>  bath tub")))
#     # print(tokenizer.encode("<scene>  bath tub"))
#     # print(tokenizer.decode(tokenizer.encode("<scene>bath tub")))
#     # print(tokenizer.encode("<scene>bath tub"))
#     # print(tokenizer.decode(tokenizer.encode("<scene> pillow")))  
#     # print(tokenizer.encode("<scene>"))
#     # print(tokenizer.decode(tokenizer.encode("<scene> ")))
#     # print(tokenizer.decode(tokenizer.encode("<scene> chair")))
#     # print(tokenizer.decode(tokenizer.encode("chair <scene>")))
#     # print(input_ids.shape)

def construct_selection_prompt(
    tokenizer,
    text_before_snapshot,
    feature_before_snapshot,
    frontier_text,
    frontier_features,
    snapshot_info_dict,
    max_length,
    prefiltering,
    # parse result of prefiltering output
    ranking,
    topk,
    num_visual_tokens,
):
    snapshot_text, snapshot_features, snapshot_index, snapshot_id_mapping = prepare_snapshot_input(
        snapshot_info_dict.seen_classes,
        snapshot_info_dict.classes,
        snapshot_info_dict.features,
        prefiltering,
        ranking,
        topk,
        num_visual_tokens,
    )
    
    text = text_before_snapshot + snapshot_text + frontier_text
    scene_feature = feature_before_snapshot + [snapshot_features] + [frontier_features]
    scene_feature = [f for f in scene_feature if f is not None]
    scene_feature = torch.cat(scene_feature, dim=0)
    # format answer
    text += "Answer: "
    # print("snapshot", tokenizer.encode("snapshot"))
    # print("placeholder", tokenizer.encode("placeholder"))
    text = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    input_ids = text["input_ids"]
    # replace the placeholder token with the snapshot token
    # snapshot_token_id = tokenizer("frame").input_ids[-1]
    # print(tokenizer("frame").input_ids)
    # print(tokenizer("<scene> / frame").input_ids)
    # print(tokenizer("frontier").input_ids)
    # print(tokenizer("<scene> / frontier").input_ids)
    # print(tokenizer.decode(8862))
    # print(tokenizer.decode(631))
    # input()
    # placeholder_token_id = 27074
    # input_ids[input_ids == placeholder_token_id] = snapshot_token_id
    length = torch.nonzero(input_ids).shape[0]

    # print('length', length)
    # print(
    #     tokenizer.decode(input_ids[0][0:length+1])
    # )
    
    attention_mask = text["attention_mask"]
    scene_token_id = tokenizer(SCENE_TOKEN).input_ids[-1]
    scene_insert_loc = (
        (input_ids == scene_token_id).nonzero()[:, 1].reshape(-1)
    )
    # print every token right after a scene token
    # print(input_ids[0][scene_insert_loc + 1])
    
    input_dict = EasyDict(
        text=text,
        input_ids=input_ids,
        length=length,
        scene_length=len(scene_feature),
        attention_mask=attention_mask,
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc,
    )
    return input_dict, snapshot_id_mapping

def collate_prefilter_wrapper(batch):
    # wrap up the prefiltering batch
    max_filter_length = max(b.filter_length for b in batch) + 1
    return EasyDict(
        # Jiachen TODO 7
        filter_input_ids=torch.cat([b.filter_input_ids for b in batch])[
            ..., :max_filter_length
        ],
        filter_attention_mask=torch.cat(
            [b.filter_attention_mask for b in batch]
        )[..., :max_filter_length],
        filter_length=torch.tensor([b.filter_length for b in batch]),
        # dummy wrapper for selection prompt
        selection_dict = [b.selection_dict for b in batch]
    )
    
def get_item(tokenizer, step_dict):
    # load a whole episode and each step within it
    step = step_dict
    # episode = step_dict['episode']
    # scene = step['scene']
    obj_map = step['obj_map']
    obj_position_map = step['obj_position_map']
 
    text_before_snapshot, feature_before_snapshot = prepare_prompt_before_snapshot(step)
    # replace scene graph in each steps with scene feature
    snapshot_features,snapshot_classes = [],[]
    snapshot_positions = []
    snapshot_index = 0
    seen_classes = set()
    #seen_objects = set()
    for i, rgb_id in enumerate(step["snapshot_features"].keys()):
        # No need to filter here (both scene_graph and snapshots objects from the json files)
        snapshot_feature = step["snapshot_features"][rgb_id]
        snapshot_class = [obj_map[int(sid)] for sid in step["snapshot_objects"][rgb_id]]
        seen_classes.update(snapshot_class)
        snapshot_classes.append(
            snapshot_class
        )
        snapshot_features.append(snapshot_feature)
        snapshot_positions.append(
            torch.mean(
                torch.tensor(
                    np.array([
                        obj_position_map[str(sid)]
                        for sid in step["snapshot_objects"][rgb_id]
                    ])
                ),
                dim=0,
            )
        )
        snapshot_index += 1

    if step.get("add_positional_encodings") is True:
        snapshot_features = [
            sum_positional_encodings(
                snapshot_features[i].unsqueeze(0),
                snapshot_positions[i],
                PE,
                num_bins=NUM_BINS,
                coord_range=COORD_RANGE,
            ).squeeze(0)
            for i in range(len(snapshot_features))
        ]

    snapshot_info_dict = EasyDict(
        seen_classes = seen_classes,
        classes = snapshot_classes,
        features = snapshot_features
    )
    
    frontier_text, frontier_features = prepare_frontier(step)
    if step.get("use_prefiltering") is True:
        # format prefiltering input
        filter_input_ids, filter_length, filter_attention_mask = prepare_prefiltering_prompt(
            step["question"],
            tokenizer,
            list(seen_classes),
            2048,
            step["top_k_categories"],
        )
        selection_dict = EasyDict(
            text_before_snapshot = text_before_snapshot,
            feature_before_snapshot = feature_before_snapshot,
            frontier_text = frontier_text,
            frontier_features = frontier_features,
            snapshot_info_dict = snapshot_info_dict,
        )
        input_dict = EasyDict(
            filter_input_ids = filter_input_ids,
            filter_length = filter_length,
            filter_attention_mask = filter_attention_mask,
            selection_dict = selection_dict,
        )
        batch = [input_dict]
        return collate_prefilter_wrapper([input_dict])
    else:
        # format selection input
        # no need use id mapping when not use prefiltering
        input_dict,_ = construct_selection_prompt(
            tokenizer,
            text_before_snapshot,
            feature_before_snapshot,
            frontier_text,
            frontier_features,
            snapshot_info_dict,
            4096,
            False,
            None,
            None,
            step["num_visual_tokens"],
        )
        batch = [input_dict]
        # print('before wrap up')
        # print(tokenizer.decode(input_dict.input_ids[input_dict.input_ids != tokenizer.pad_token_id]))
        return collate_wrapper(batch)
