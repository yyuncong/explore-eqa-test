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
def prepare_prompt_before_object(step):
    
    text =  f"Question: {step['question']}\n" 
    multi_src_feature = []
    if step.get("use_egocentric_views") is True:
        text += "Followings are the egocentric views:\n "
        for i in range(len(step["egocentric_view_features"])):
            text += f"<scene> "
        text += "/\n"
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
    
    text += f"Select the frontier/object that would help finding the answer of the question.\n"

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
    
    text = "Below are all the frontiers that we can explore:\n"
    if len(step['frontiers']) > 0:
        for i, frontier in enumerate(step['frontiers']):
            text += f"frontier {i} <scene> "
    else:
        text += f"No frontier available "
    text += "/\n"
    frontier_features = step["frontier_features"]
    if len(step['frontiers']) > 0 and step.get("add_positional_encodings") is True:
        frontier_positions = torch.tensor(step["frontier_positions"])
        frontier_features = sum_positional_encodings(
            frontier_features, frontier_positions, PE, num_bins=NUM_BINS, coord_range=COORD_RANGE
        )
    
    return text, frontier_features
    
def prepare_object_input(
    class2object,
    object_classes,
    object_features,
    prefiltering,
    ranking,
    topk,
):
    object_index = len(object_classes)
    # the mapping from transformed object index to original object index(used by tsdf)
    object_id_mapping = None
    if prefiltering:
        ranking = [cls for cls in ranking if cls in class2object.keys()]
        ranking = ranking[:topk]
        object_classes = [cls for cls in ranking for _ in class2object[cls]]
        object_features = [
            object_features[obj_idx] for cls in ranking for obj_idx in class2object[cls]
        ]
        object_id_mapping = [obj_idx for cls in ranking for obj_idx in class2object[cls]]
        # Note that if apply prefiltering, we may have #(objects) < object_index
        # 4. reassign object_index = #(object)
        object_index = len(object_classes)

    text = "These are the objects already in our scene graph:\n"
    for i, class_name in enumerate(object_classes):
        text += f"object {i} {class_name} <scene> "

    if object_index == 0:
        text += f"No object available "
        # construct zero scene feature if all objects are missed
        object_features = None
    else:
        object_features = torch.stack(object_features, dim=0)
    text += "/\n"
    #print("object prompt \n", text)
    return text, object_features, object_index, object_id_mapping

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

def construct_selection_prompt(
    tokenizer,
    text_before_object,
    feature_before_object,
    frontier_text,
    frontier_features,
    # dict object contains object features/predictions/classes
    object_info_dict,
    max_length,
    prefiltering,
    # parse result of prefiltering output
    ranking,
    topk
):
    object_text, object_features, object_index, object_id_mapping = prepare_object_input(
        object_info_dict.class2object,
        object_info_dict.classes,
        object_info_dict.features,
        prefiltering,
        ranking,
        topk
    )
    
    text = text_before_object + object_text + frontier_text
    scene_feature = feature_before_object + [object_features] + [frontier_features]
    scene_feature = [f for f in scene_feature if f is not None]
    scene_feature = torch.cat(scene_feature, dim=0)
    # format answer
    text += "Answer: "
    # print("final selection prompt \n", text)
    
    text = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length"
    )
    input_ids = text["input_ids"]
    length = torch.nonzero(input_ids).shape[0]

    attention_mask = text["attention_mask"]
    scene_token_id = tokenizer(SCENE_TOKEN).input_ids[-1]
    scene_insert_loc = (
        (input_ids == scene_token_id).nonzero()[:, 1].reshape(-1)
    )
    
    input_dict = EasyDict(
        text=text,
        input_ids=input_ids,
        length=length,
        scene_length=len(scene_feature),
        attention_mask=attention_mask,
        scene_feature=scene_feature,
        scene_insert_loc=scene_insert_loc,
    )
    return input_dict, object_id_mapping

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
    scene = step['scene']
    scene_feature_map = step['scene_feature_map']
    obj_map = step['obj_map']
    obj_position_map = step['obj_position_map']
    '''
    text_before_object =  f"Question: {step['question']}\n" 

    if step.get("use_egocentric_views") is True:
        text_before_object += "Followings are the egocentric views:\n "
        for i in range(len(step["egocentric_view_features"])):
            text_before_object += f"<scene> "
        text_before_object += "/\n"
        egocentric_features = step["egocentric_view_features"]
    
    text_before_object += f"Select the frontier/object that would help finding the answer of the question.\n"

    if step.get("use_action_memory") is True:
        text += f"Here is your selection in the previous step:\n "
        if step["memory_feature"] is None:
            text += f"No selection in the previous step. "
        else:
            text += f"<scene> "
        memory_feature = step["memory_feature"]
        text += "/\n"
    '''
    text_before_object, feature_before_object = prepare_prompt_before_object(step)
    # replace scene graph in each steps with scene feature
    object_features,object_classes = [],[]
    object_positions = []
    object_index = 0
    class2object = defaultdict(list)
    '''
    for i, sid in enumerate(step["scene_graph"]):
        if str(sid) not in scene_feature_map.keys() or sid not in obj_map.keys():
            continue
        else:
            object_feature = scene_feature_map[str(sid)]
            object_features.append(object_feature)
            class_name = obj_map[sid]
            text += f"object {object_index} {class_name} <scene> "
            object_index += 1
    if object_index == 0:
        text += f"No object available "
    text += "/\n"
    '''
    for i, sid in enumerate(step["scene_graph"]):
        if str(sid) not in scene_feature_map.keys() or sid not in obj_map.keys():
            continue
        else:
            try:
                object_feature = scene_feature_map[str(sid)]
                object_classes.append(obj_map[sid])
                object_features.append(object_feature)
                # object_positions.append(obj_position_map[sid])
                object_positions.append(obj_position_map[str(sid)])
                class2object[obj_map[sid]].append(object_index)
                object_index += 1
            except:
                continue
    print("length object features", len(object_features))
    if step.get("add_positional_encodings") is True:
        object_features = [
            sum_positional_encodings(
                object_features[i].unsqueeze(0), 
                object_positions[i],
                PE, 
                num_bins=NUM_BINS, 
                coord_range=COORD_RANGE
            ).squeeze(0)
            for i in range(len(object_features))
        ]

    '''
    if len(object_features) == 0:
        # construct zero scene feature if all objects are missed
        object_features = None
    else:
        object_features = torch.stack(object_features, dim = 0)
    '''
    object_info_dict = EasyDict(
        class2object = class2object,
        classes = object_classes,
        features = object_features
    )
    '''
    text += "Below are all the frontiers that we can explore:\n"
    if len(step['frontiers']) > 0:
        for i, frontier in enumerate(step['frontiers']):
            text += f"frontier {i} <scene> "
    else:
        text += f"No frontier available "
    text += "/\n"
    frontier_features = step["frontier_features"]
    text += "Answer: "
    '''
    frontier_text, frontier_features = prepare_frontier(step)
    if step.get("use_prefiltering") is True:
        # format prefiltering input
        filter_input_ids, filter_length, filter_attention_mask = prepare_prefiltering_prompt(
            step["question"],
            tokenizer,
            list(class2object.keys()),
            1024,
            step["top_k_categories"],
        )
        selection_dict = EasyDict(
            text_before_object = text_before_object,
            feature_before_object = feature_before_object,
            frontier_text = frontier_text,
            frontier_features = frontier_features,
            object_info_dict = object_info_dict,
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
            text_before_object,
            feature_before_object,
            frontier_text,
            frontier_features,
            object_info_dict,
            1024,
            False,
            None,
            None
        )
        '''
        scene_feature = feature_before_object + [object_features] + feature_after_object
        scene_feature = [f for f in scene_feature if f is not None]
        scene_feature = torch.cat(scene_feature, dim=0)
        step["scene_feature"] = scene_feature
        # remove scene graph id --- remove this if we need to keep id
        print(text)
    
        text = tokenizer(text, return_tensors = "pt",
                            max_length = 1024,
                            truncation = True,
                            padding = 'max_length')
        input_ids = text["input_ids"]
        length = torch.nonzero(input_ids).shape[0]
        
        attention_mask = text["attention_mask"]
        
        # only pick the index of the first occured token
        scene_token_id = tokenizer(SCENE_TOKEN).input_ids[-1]
        scene_insert_loc = (input_ids == scene_token_id).nonzero()[:,1].reshape(-1)
                    
        batch = [
            EasyDict(
                text = text,
                input_ids = input_ids,
                length = length,
                scene_length = len(scene_feature),
                attention_mask = attention_mask,
                scene_feature = scene_feature,
                scene_insert_loc = scene_insert_loc,
            )
        ]
        '''
        batch = [input_dict]
        return collate_wrapper(batch)
