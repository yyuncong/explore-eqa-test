import matplotlib.pyplot as plt
import matplotlib.image

import os
import random
os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np
import torch
import math
from PIL import Image

np.set_printoptions(precision=3)
import json
import logging
import glob
import open_clip
from ultralytics import YOLOWorld, SAM
from hydra import initialize, compose
from src.habitat import (
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_new_cg import TSDFPlanner, Frontier, SnapShot
from src.scene import Scene
from src.eval_utils_snapshot_new import (
    prepare_step_dict, 
    get_item, 
    encode, 
    load_scene_features, 
    rgba2rgb, 
    load_checkpoint, 
    collate_wrapper, 
    construct_selection_prompt,
    merge_patches,
    load_ds_checkpoint
)
from src.eval_utils_snapshot_new import SCENE_TOKEN

from conceptgraph.utils.general_utils import measure_time

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from easydict import EasyDict


def infer_prefilter(model, tokenizer, sample):
    # return prefiltered object list
    filter_input_ids = sample.filter_input_ids.to("cuda")
    if len(torch.where(sample.filter_input_ids==22550)[1]) == 0:
        logging.info(f"invalid: no token 'answer'!")
        return None
    answer_ind = torch.where(sample.filter_input_ids==22550)[1][0].item()
    filter_input_ids = filter_input_ids[:, :answer_ind+2]
    '''
    logging.info('prefiltering prompt')
    logging.info(
        tokenizer.decode(filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id])
    )
    '''
    with torch.no_grad():
        with torch.inference_mode() and torch.autocast(device_type="cuda"):
            filter_output_ids = model.generate(
                filter_input_ids,
                feature_dict=None,
                do_sample=False,
                max_new_tokens=100,
            )
    # parse the prefilter output
        filter_outputs = tokenizer.decode(filter_output_ids[0, filter_input_ids.shape[1]:]).replace("</s>", "").strip()
    #print("the output of prefiltering", filter_outputs)
    # logging.info(f"prefiltering output: {filter_outputs}")
    if filter_outputs == "No object available":
        return []
    else:
        filter_outputs = filter_outputs.split("\n")
        # print("parsed output of prefiltering", filter_outputs)
        return filter_outputs

def infer_selection(model, tokenizer, sample):
    feature_dict = EasyDict(
        scene_feature = sample.scene_feature.to("cuda"),
        scene_insert_loc = sample.scene_insert_loc,
        scene_length = sample.scene_length,
    )
    input_ids = sample.input_ids.to("cuda")
    '''
    logging.info('final input to the model')
    logging.info(
        tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id])
    )
    '''
    
    # input()
    # the loss of : exists in infer_selection
    # but in final prompt
    if len(torch.where(sample.input_ids==22550)[1]) == 0:
        logging.info(f"invalid: no token 'answer'!")
        return None
        eixt(0)
    answer_ind = torch.where(sample.input_ids==22550)[1][0].item()
    input_ids = input_ids[:, :answer_ind+2]
    with torch.no_grad():
        with torch.inference_mode() and torch.autocast(device_type="cuda"):
            output_ids = model.generate(
                input_ids,
                feature_dict=feature_dict,
                do_sample=False,
                max_new_tokens=10,
            )
        outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).replace("</s>", "").strip()
    return outputs

def inference(model, tokenizer, step_dict, cfg):
    step_dict["use_prefiltering"] = cfg.prefiltering
    #step_dict["use_egocentric_views"] = cfg.egocentric_views
    #step_dict["use_action_memory"] = cfg.action_memory
    step_dict["top_k_categories"] = cfg.top_k_categories
    step_dict["add_positional_encodings"] = cfg.add_positional_encodings

    num_visual_tokens = (cfg.visual_feature_size // cfg.patch_size) ** 2
    step_dict["num_visual_tokens"] = num_visual_tokens
    # print("pos", step_dict["add_positional_encodings"])
    # try:
    sample = get_item(
        tokenizer, step_dict
    )
    # except:
    #     logging.info(f"Get item failed! (most likely no frontiers and no objects)")
    #     return None

    if cfg.prefiltering:
        filter_outputs = infer_prefilter(model,tokenizer,sample)
        if filter_outputs is None:
            return None
        selection_dict = sample.selection_dict[0]
        selection_input, snapshot_id_mapping = construct_selection_prompt(
            tokenizer, 
            selection_dict.text_before_snapshot,
            selection_dict.feature_before_snapshot,
            selection_dict.frontier_text,
            selection_dict.frontier_features,
            selection_dict.snapshot_info_dict,
            4096,
            True,
            filter_outputs,
            cfg.top_k_categories,
            num_visual_tokens
        )
        sample = collate_wrapper([selection_input])
        outputs = infer_selection(model,tokenizer,sample)
        return outputs, snapshot_id_mapping
    else:
        # already loss Answer/:
        #print('before input into inference')
        #print(
        #    tokenizer.decode(sample.input_ids[0][sample.input_ids[0] != tokenizer.pad_token_id])
        #)
        outputs = infer_selection(model,tokenizer,sample)
        return outputs


def main(cfg):
    # use hydra to load concept graph related configs
    with initialize(config_path="conceptgraph/hydra_configs", job_name="app"):
        cfg_cg = compose(config_name=cfg.concept_graph_config_name)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    all_questions_list = os.listdir(cfg.path_data_dir)
    all_questions_list = [question_id for question_id in all_questions_list if 800 <= int(question_id.split('-')[0])]
    total_questions = len(all_questions_list)
    all_scene_list = sorted(
        list(set(
            [question_id.split('_')[0] for question_id in all_questions_list]
        ))
    )
    random.shuffle(all_scene_list)
    logging.info(f"Loaded {len(all_questions_list)} questions in {len(all_scene_list)} scenes.")

    ## Initialize the detection models
    detection_model = YOLOWorld(cfg.yolo_model_name)
    logging.info(f"Load YOLO model {cfg.yolo_model_name} successful!")

    sam_predictor = SAM('sam_l.pt')  # SAM('sam_l.pt') # UltraLytics SAM
    # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM

    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(cfg_cg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    logging.info(f"Load VLM!")
    # Initialize LLaVA model
    # model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/work/pi_chuangg_umass_edu/yuncong/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=None, add_multisensory_token=True
    )
    # model = model.to("cuda")
    if not cfg.use_deepspeed:
        load_checkpoint(model, cfg.model_path)
    else:
        load_ds_checkpoint(model, cfg.model_path, exclude_frozen_parameters=True)
    model = model.to("cuda")
    # model = None
    model.eval()
    logging.info(f"Load VLM successful!")

    # for each scene, answer each question
    question_ind = 0
    success_count = 0
    same_class_count = 0
    wrong_snapshot_count = 0
    too_many_steps_count = 0
    other_errors_count = 0

    success_list = []
    path_length_list = []
    dist_from_chosen_to_target_list = []

    for scene_id in all_scene_list:
        all_question_id_in_scene = [q for q in all_questions_list if scene_id in q]

        ##########################################################
        # rand_q = np.random.randint(0, len(all_questions_in_scene) - 1)
        # all_questions_in_scene = all_questions_in_scene[rand_q:rand_q+1]
        # all_questions_in_scene = [q for q in all_questions_in_scene if '00324' in q['question_id']]
        # if len(all_questions_in_scene) == 0:
        #     continue
        # random.shuffle(all_question_id_in_scene)
        # all_question_id_in_scene = all_question_id_in_scene[:2]
        # all_questions_in_scene = [q for q in all_questions_in_scene if "00109" in q['question_id']]
        ##########################################################

        bbox_data_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
        if not os.path.exists(bbox_data_path):
            logging.info(f"Scene {scene_id} bbox data not found, skip")
            continue
        bounding_box_data = json.load(open(bbox_data_path, "r"))

        # load scene
        try:
            del scene
        except:
            pass

        scene = Scene(scene_id, cfg, cfg_cg)

        # Set the classes for the detection model
        detection_model.set_classes(scene.obj_classes.get_classes_arr())

        # load question files
        question_file = json.load(open(cfg.question_file_path, "r"))
        question_file = {item['question_id']: item for item in question_file}

        # Evaluate each question
        for question_id in all_question_id_in_scene:
            question_ind += 1
            metadata = json.load(open(os.path.join(cfg.path_data_dir, question_id, "metadata.json"), "r"))

            # load question data
            question = metadata["question"]
            answer = metadata["answer"]
            init_pts = metadata["init_pts"]
            init_angle = metadata["init_angle"]
            target_obj_id = metadata['target_obj_id']
            target_obj_class = metadata['target_obj_class']
            target_obs_pos = question_file[question_id.split('_path')[0]]['position']
            # get target object global location
            obj_bbox = [item['bbox'] for item in bounding_box_data if int(item['id']) == target_obj_id][0]
            obj_bbox = np.asarray(obj_bbox)  # (2, 3)
            obj_bbox_center = np.mean(obj_bbox, axis=0)
            obj_bbox_center = obj_bbox_center[[0, 2, 1]]

            episode_data_dir = os.path.join(str(cfg.output_dir), question_id)
            episode_object_observe_dir = os.path.join(episode_data_dir, 'object_observations')
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_object_observe_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)
            os.makedirs(episode_snapshot_dir, exist_ok=True)

            pts = np.asarray(init_pts)
            angle = init_angle
            rotation = get_quaternion(angle, 0)

            # initialize the TSDF
            pts_normal = pos_habitat_to_normal(pts)
            floor_height = pts_normal[-1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            logging.info(
                f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
            )
            try:
                del tsdf_planner
            except:
                pass
            tsdf_planner = TSDFPlanner(
                vol_bnds=tsdf_bnds,
                voxel_size=cfg.tsdf_grid_size,
                floor_height_offset=0,
                pts_init=pts_normal,
                init_clearance=cfg.init_clearance * 2,
            )

            target_center_voxel = tsdf_planner.world2vox(pos_habitat_to_normal(obj_bbox_center))
            # record the history of the agent's path
            pts_pixs = np.empty((0, 2))
            pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

            # clear up the previous detected frontiers and objects
            scene.clear_up_detections()

            logging.info(f'\n\nQuestion id {question_id} initialization successful!')

            # run steps
            target_found = False
            explore_dist = 0.0
            cnt_step = -1
            memory_feature = None
            dist_from_chosen_to_target = None
            target_obj_id_det_list = []  # record all the detected target object ids, and use the most frequent one as the final target object id

            all_snapshot_features = {}

            while cnt_step < num_step - 1:
                cnt_step += 1
                logging.info(f"\n== step: {cnt_step}")
                step_dict = {}
                angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                total_views = 1 + cfg.extra_view_phase_1
                all_angles = [angle + angle_increment * (i - total_views // 2) for i in range(total_views)]
                # let the main viewing angle be the last one to avoid potential overwriting problems
                main_angle = all_angles.pop(total_views // 2)
                all_angles.append(main_angle)

                # observe and update the TSDF
                rgb_egocentric_views_features = []
                all_added_obj_ids = []
                for view_idx, ang in enumerate(all_angles):
                    obs, cam_pose = scene.get_observation(pts, ang)
                    rgb = obs["color_sensor"]
                    depth = obs["depth_sensor"]
                    semantic_obs = obs["semantic_sensor"]
                    rgb = rgba2rgb(rgb)

                    cam_pose_normal = pose_habitat_to_normal(cam_pose)
                    cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                    obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                    with torch.no_grad():
                        annotated_rgb, added_obj_ids, target_obj_id_det = scene.update_scene_graph(
                            image_rgb=rgb[..., :3], depth=depth, intrinsics=cam_intr, cam_pos=cam_pose,
                            detection_model=detection_model, sam_predictor=sam_predictor, clip_model=clip_model,
                            clip_preprocess=clip_preprocess, clip_tokenizer=clip_tokenizer,
                            pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
                            img_path=obs_file_name,
                            frame_idx=cnt_step * total_views + view_idx,
                            target_obj_mask=semantic_obs == target_obj_id,
                        )
                        img_feature = encode(model, image_processor, rgb).mean(0)
                        img_feature = merge_patches(
                            img_feature.view(cfg.visual_feature_size, cfg.visual_feature_size, -1),
                            cfg.patch_size
                        )
                        all_snapshot_features[obs_file_name] = img_feature.to("cpu")
                        rgb_egocentric_views_features.append(img_feature.to("cpu"))
                        if cfg.save_visualization or cfg.save_frontier_video:
                            plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), annotated_rgb)
                        if target_obj_id_det is not None:
                            target_obj_id_det_list.append(target_obj_id_det)
                        all_added_obj_ids += added_obj_ids

                    # clean up or merge redundant objects periodically
                    scene.periodic_cleanup_objects(frame_idx=cnt_step * total_views + view_idx, pts=pts)

                    # TODO: not sure whether doing this is correct when solving the problem that target_obj_id_det is merged into another object in periodic_cleanup_objects, resulting in the target_obj_id_det not in the scene.objects
                    target_obj_id_det_list_filtered = [obj_id for obj_id in target_obj_id_det_list if obj_id in scene.objects]
                    if len(target_obj_id_det_list_filtered) != len(target_obj_id_det_list):
                        logging.info(f"!!!!!!!!! {target_obj_id_det_list} -> {target_obj_id_det_list_filtered}")
                        target_obj_id_det_list = target_obj_id_det_list_filtered

                    # TSDF fusion
                    tsdf_planner.integrate(
                        color_im=rgb,
                        depth_im=depth,
                        cam_intr=cam_intr,
                        cam_pose=cam_pose_tsdf,
                        obs_weight=1.0,
                        margin_h=int(cfg.margin_h_ratio * img_height),
                        margin_w=int(cfg.margin_w_ratio * img_width),
                        explored_depth=cfg.explored_depth,
                    )

                # cluster all the newly added objects
                all_added_obj_ids = [obj_id for obj_id in all_added_obj_ids if obj_id in scene.objects]
                # as well as the objects nearby
                for obj_id, obj in scene.objects.items():
                    if np.linalg.norm(obj['bbox'].center[[0, 2]] - pts[[0, 2]]) < cfg.scene_graph.obj_include_dist + 0.5:
                        all_added_obj_ids.append(obj_id)
                scene.update_snapshots(obj_ids=set(all_added_obj_ids))
                logging.info(f"Step {cnt_step} {len(scene.objects)} objects, {len(scene.snapshots)} snapshots")

                # update the mapping of object id to class name, since the objects have been updated
                object_id_to_name = {obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()}

                step_dict["snapshot_features"] = {}
                step_dict["snapshot_objects"] = {}
                for rgb_id, snapshot in scene.snapshots.items():
                    step_dict["snapshot_features"][rgb_id] = all_snapshot_features[rgb_id]
                    step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
                # print(step_dict["snapshot_objects"])
                # input()

                # record current scene graph
                step_dict["scene_graph"] = list(scene.objects.keys())
                step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]
                step_dict["obj_map"] = object_id_to_name
                step_dict["position"] = np.array(pts)[None,]
                obj_positions_map = {
                    str(obj_id): obj['bbox'].center
                    for obj_id, obj in scene.objects.items()
                }
                obj_positions_map = {
                    key: value - step_dict["position"] for key, value in obj_positions_map.items()
                }
                step_dict["obj_position_map"] = obj_positions_map

                update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
                if not update_success:
                    logging.info("Warning! Update frontier map failed!")

                if target_found:
                    break

                # Get observations for each frontier and store them
                for i, frontier in enumerate(tsdf_planner.frontiers):
                    pos_voxel = frontier.position
                    pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                    pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                    assert (frontier.image is None and frontier.feature is None) or (frontier.image is not None and frontier.feature is not None), f"{frontier.image}, {frontier.feature is None}"
                    # Turn to face the frontier point
                    if frontier.image is None:
                        view_frontier_direction = np.asarray([pos_world[0] - pts[0], 0., pos_world[2] - pts[2]])

                        obs = scene.get_frontier_observation(pts, view_frontier_direction)
                        frontier_obs = obs["color_sensor"]

                        if cfg.save_frontier_video or cfg.save_visualization:
                            plt.imsave(
                                os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                                frontier_obs,
                            )
                        processed_rgb = rgba2rgb(frontier_obs)
                        with torch.no_grad():
                            img_feature = encode(model, image_processor, processed_rgb).mean(0)
                        img_feature = merge_patches(
                            img_feature.view(cfg.visual_feature_size, cfg.visual_feature_size, -1),
                            cfg.patch_size
                        )
                        assert img_feature is not None
                        frontier.image = f"{cnt_step}_{i}.png"
                        frontier.feature = img_feature

                if cfg.choose_every_step:
                    if tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier:
                        # reset target point to allow the model to choose again
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None

                logging.info(f"Target object detection list: {target_obj_id_det_list}")

                # use the most common id in the list as the target object id
                if len(target_obj_id_det_list) > 0:
                    target_obj_id_estimate = max(set(target_obj_id_det_list), key=target_obj_id_det_list.count)
                    logging.info(f"Estimated target object id: {target_obj_id_estimate} {object_id_to_name[target_obj_id_estimate]}")
                else:
                    target_obj_id_estimate = -1

                if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                    # choose a frontier, and set it as the explore target
                    step_dict["frontiers"] = []
                    # since we skip the stuck frontier for input of the vlm, we need to map the
                    # vlm output frontier id to the tsdf planner frontier id
                    ft_id_to_vlm_id = {}
                    vlm_id_count = 0
                    for i, frontier in enumerate(tsdf_planner.frontiers):
                        frontier_dict = {}
                        pos_voxel = frontier.position
                        pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                        pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                        frontier_dict["coordinate"] = pos_world.tolist()
                        assert frontier.image is not None and frontier.feature is not None
                        frontier_dict["rgb_feature"] = frontier.feature
                        frontier_dict["rgb_id"] = frontier.image

                        step_dict["frontiers"].append(frontier_dict)

                        ft_id_to_vlm_id[i] = vlm_id_count
                        vlm_id_count += 1
                    vlm_id_to_ft_id = {v: k for k, v in ft_id_to_vlm_id.items()}

                    if cfg.egocentric_views:
                        assert len(rgb_egocentric_views_features) == total_views
                        rgb_egocentric_views_features = torch.cat(rgb_egocentric_views_features, dim=0)
                        step_dict["egocentric_view_features"] = rgb_egocentric_views_features
                        step_dict["use_egocentric_views"] = True

                    if cfg.action_memory:
                        step_dict["memory_feature"] = memory_feature
                        step_dict["use_action_memory"] = True

                    # add model prediction here
                    if len(step_dict["frontiers"]) > 0:
                        step_dict["frontier_features"] = torch.cat(
                            [
                                frontier["rgb_feature"] for frontier in step_dict["frontiers"]
                            ],
                            dim=0
                        ).to("cpu")
                        step_dict["frontier_positions"] = np.array(
                            [f["coordinate"] for f in step_dict["frontiers"]]
                        ) - step_dict["position"]
                    else:
                        step_dict["frontier_features"] = None
                        step_dict["frontier_positions"] = None
                    step_dict["question"] = question
                    step_dict["scene"] = scene_id
                    if cfg.prefiltering:
                        outputs, snapshot_id_mapping = inference(model, tokenizer, step_dict, cfg)
                    else:
                        outputs = inference(model, tokenizer, step_dict, cfg)
                        snapshot_id_mapping = None
                    if outputs is None:
                        # encounter generation error
                        logging.info(f"Question id {question_id} invalid: model generation error!")
                        break
                    try:
                        target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
                        #print(f"Prediction: {target_type}, {target_index}")
                        logging.info(f"Prediction: {target_type}, {target_index}")
                    except:
                        logging.info(f"Wrong output format, failed!")
                        break

                    if target_type not in ["snapshot", "frontier"]:
                        logging.info(f"Invalid prediction type: {target_type}, failed!")
                        print(target_type)
                        break

                    if target_type == "snapshot":
                        # TODO: the problem needed to be fixed here
                        if snapshot_id_mapping is not None:
                            if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
                                logging.info(f"target index can not match real objects: {target_index}, failed!")
                                break
                            target_index = snapshot_id_mapping[int(target_index)]
                            logging.info(f"The index of target snapshot {target_index}")
                        if int(target_index) < 0 or int(target_index) >= len(scene.objects):
                            logging.info(f"Prediction out of range: {target_index}, {len(scene.objects)}, failed!")
                            break
                        pred_target_snapshot = list(scene.snapshots.values())[int(target_index)]
                        logging.info(
                            "pred_target_class: " + str(' '.join([object_id_to_name[obj_id] for obj_id in pred_target_snapshot.cluster]))
                        )

                        logging.info(f"Next choice Snapshot of {pred_target_snapshot.image}")
                        tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                        # TODO: where to go if snapshot?
                        max_point_choice = pred_target_snapshot
                    else:
                        target_index = int(target_index)
                        if target_index not in vlm_id_to_ft_id.keys():
                            logging.info(f"Predicted frontier index invalid: {target_index}, failed!")
                            break
                        target_index = vlm_id_to_ft_id[target_index]
                        target_point = tsdf_planner.frontiers[target_index].position
                        logging.info(f"Next choice: Frontier at {target_point}")
                        tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
                        max_point_choice = tsdf_planner.frontiers[target_index]

                        # TODO: modify this: update memory feature only in frontiers (for now)
                        memory_feature = tsdf_planner.frontiers[target_index].feature.to("cpu")

                    if max_point_choice is None:
                        logging.info(f"Question id {question_id} invalid: no valid choice!")
                        break

                    if type(max_point_choice) == SnapShot and cfg.not_observe_snapshot:
                        # then just check the correctness of the snapshot and finish this question
                        # save the snapshot as the observation
                        os.system(f"cp {os.path.join(episode_snapshot_dir, max_point_choice.image)} {os.path.join(episode_object_observe_dir, 'target.png')}")

                        # check whether the target object is in the selected snapshot
                        if target_obj_id_estimate in max_point_choice.cluster:
                            logging.info(f"{target_obj_id_estimate} in Chosen snapshot {max_point_choice.image}! Success!")
                            target_found = True
                        else:
                            logging.info(f"Question id {question_id} choose the wrong snapshot! Failed!")

                        # check whether the class of the target object is the same as the class of the selected snapshot
                        for ss_obj_id in max_point_choice.cluster:
                            if object_id_to_name[ss_obj_id] == target_obj_class:
                                same_class_count += 1
                                break

                        # get the distance between current position to target observation position
                        dist_from_chosen_to_target = np.linalg.norm(np.asarray(pts) - np.asarray(target_obs_pos))
                        break

                    update_success = tsdf_planner.set_next_navigation_point(
                        choice=max_point_choice,
                        pts=pts_normal,
                        objects=scene.objects,
                        cfg=cfg.planner,
                        pathfinder=scene.pathfinder,
                    )
                    if not update_success:
                        logging.info(f"Question id {question_id} invalid: set_next_navigation_point failed!")
                        break

                return_values = tsdf_planner.agent_step(
                    pts=pts_normal,
                    angle=angle,
                    objects=scene.objects,
                    snapshots=scene.snapshots,
                    pathfinder=scene.pathfinder,
                    cfg=cfg.planner,
                    path_points=None,
                    save_visualization=cfg.save_visualization,
                )
                if return_values[0] is None:
                    logging.info(f"Question id {question_id} invalid: agent_step failed!")
                    break
                pts_normal, angle, pts_pix, fig, _, target_arrived = return_values

                # sanity check
                obj_exclude_count = sum([1 if obj['num_detections'] < 2 else 0 for obj in scene.objects.values()])
                total_objs_count = sum(
                    [len(snapshot.cluster) for snapshot in scene.snapshots.values()]
                )
                assert len(scene.objects) == total_objs_count + obj_exclude_count, f"{len(scene.objects)} != {total_objs_count} + {obj_exclude_count}"
                total_objs_count = sum(
                    [len(set(snapshot.cluster)) for snapshot in scene.snapshots.values()]
                )
                assert len(scene.objects) == total_objs_count + obj_exclude_count, f"{len(scene.objects)} != {total_objs_count} + {obj_exclude_count}"
                for obj_id in scene.objects.keys():
                    exist_count = 0
                    for ss in scene.snapshots.values():
                        if obj_id in ss.cluster:
                            exist_count += 1
                    if scene.objects[obj_id]['num_detections'] < 2:
                        assert exist_count == 0, f"{exist_count} != 0 for obj_id {obj_id}, {scene.objects[obj_id]['class_name']}"
                    else:
                        assert exist_count == 1, f"{exist_count} != 1 for obj_id {obj_id}, {scene.objects[obj_id]['class_name']}"
                for ss in scene.snapshots.values():
                    assert len(ss.cluster) == len(set(ss.cluster)), f"{ss.cluster} has duplicates"
                    assert len(ss.full_obj_list.keys()) == len(set(ss.full_obj_list.keys())), f"{ss.full_obj_list.keys()} has duplicates"
                    for obj_id in ss.cluster:
                        assert obj_id in ss.full_obj_list, f"{obj_id} not in {ss.full_obj_list.keys()}"
                    for obj_id in ss.full_obj_list.keys():
                        assert obj_id in scene.objects, f"{obj_id} not in scene objects"
                # check whether the snapshots in scene.snapshots and scene.frames are the same
                for file_name, ss in scene.snapshots.items():
                    assert ss.cluster == scene.frames[file_name].cluster, f"{ss}\n!=\n{scene.frames[file_name]}"
                    assert ss.full_obj_list == scene.frames[file_name].full_obj_list, f"{ss}\n==\n{scene.frames[file_name]}"

                # update the agent's position record
                pts_pixs = np.vstack((pts_pixs, pts_pix))
                if cfg.save_visualization:
                    # Add path to ax5, with colormap to indicate order
                    visualization_path = os.path.join(episode_data_dir, "visualization")
                    os.makedirs(visualization_path, exist_ok=True)
                    ax5 = fig.axes[4]
                    ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                    ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)

                    # add target object bbox
                    color = 'green' if len(target_obj_id_det_list) > 0 else 'red'
                    ax5.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax1, ax2, ax4 = fig.axes[0], fig.axes[1], fig.axes[3]
                    ax4.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax1.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)
                    ax2.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s=120)

                    fig.tight_layout()
                    plt.savefig(os.path.join(visualization_path, "{}_map.png".format(cnt_step)))
                    plt.close()

                if cfg.save_frontier_video:
                    frontier_video_path = os.path.join(episode_data_dir, "frontier_video")
                    os.makedirs(frontier_video_path, exist_ok=True)
                    num_images = len(tsdf_planner.frontiers)
                    if type(max_point_choice) == SnapShot:
                        num_images += 1
                    side_length = int(np.sqrt(num_images)) + 1
                    side_length = max(2, side_length)
                    fig, axs = plt.subplots(side_length, side_length, figsize=(20, 20))
                    for h_idx in range(side_length):
                        for w_idx in range(side_length):
                            axs[h_idx, w_idx].axis('off')
                            i = h_idx * side_length + w_idx
                            if (i < num_images - 1) or (i < num_images and type(max_point_choice) == Frontier):
                                img_path = os.path.join(episode_frontier_dir, tsdf_planner.frontiers[i].image)
                                img = matplotlib.image.imread(img_path)
                                axs[h_idx, w_idx].imshow(img)
                                if type(max_point_choice) == Frontier and max_point_choice.image == tsdf_planner.frontiers[i].image:
                                    axs[h_idx, w_idx].set_title('Chosen')
                            elif i == num_images - 1 and type(max_point_choice) == SnapShot:
                                img_path = os.path.join(episode_snapshot_dir, max_point_choice.image)
                                img = matplotlib.image.imread(img_path)
                                axs[h_idx, w_idx].imshow(img)
                                axs[h_idx, w_idx].set_title('Snapshot Chosen')
                    global_caption = f"{question}\n{answer}"
                    fig.suptitle(global_caption, fontsize=16)
                    plt.tight_layout(rect=(0., 0., 1., 0.95))
                    plt.savefig(os.path.join(frontier_video_path, f'{cnt_step}.png'))
                    plt.close()

                # update position and rotation
                pts_normal = np.append(pts_normal, floor_height)
                pts = pos_normal_to_habitat(pts_normal)
                rotation = get_quaternion(angle, 0)
                if type(max_point_choice) == Frontier:
                    # count the explore distance only when the agent is exploring, not approaching the target
                    explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                logging.info(f"Current position: {pts}, {explore_dist:.3f}")

                if type(max_point_choice) == SnapShot and target_arrived:
                    # get an observation and break
                    obs, _ = scene.get_observation(pts, angle)
                    rgb = obs["color_sensor"]

                    plt.imsave(os.path.join(episode_object_observe_dir, f"target.png"), rgb)

                    # get some statistics

                    # check whether the target object is in the selected snapshot
                    if target_obj_id_estimate in max_point_choice.cluster:
                        logging.info(f"{target_obj_id_estimate} in Chosen snapshot {max_point_choice.image}! Success!")
                        target_found = True
                    else:
                        logging.info(f"Question id {question_id} choose the wrong snapshot! Failed!")

                    # check whether the class of the target object is the same as the class of the selected snapshot
                    for ss_obj_id in max_point_choice.cluster:
                        if object_id_to_name[ss_obj_id] == target_obj_class:
                            same_class_count += 1
                            logging.info(f"Same class: {object_id_to_name[ss_obj_id]}")
                            break

                    # get the distance between current position to target observation position
                    dist_from_chosen_to_target = np.linalg.norm(pts - target_obs_pos)

                    break

            if target_found:
                success_count += 1
                success_list.append(1)
                logging.info(f"Question id {question_id} finish with {cnt_step} steps, {explore_dist} length")
            else:
                success_list.append(0)
                logging.info(f"Question id {question_id} failed, {explore_dist} length")

                if dist_from_chosen_to_target is not None:
                    # meaning that there is a snapshot chosen but the target object is not in the snapshot
                    wrong_snapshot_count += 1
                elif cnt_step >= num_step - 1:
                    too_many_steps_count += 1
                else:
                    other_errors_count += 1

            path_length_list.append(explore_dist)
            if dist_from_chosen_to_target is not None:
                dist_from_chosen_to_target_list.append(dist_from_chosen_to_target)

            logging.info(f"\n#######################################################")
            logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")
            logging.info(f"Same class ratio: {same_class_count}/{question_ind}")
            logging.info(f"Mean path length for success exploration: {np.mean([x for i, x in enumerate(path_length_list) if success_list[i] == 1])}")
            logging.info(f"Mean path length for all exploration: {np.mean(path_length_list)}")
            logging.info(f"Mean distance from final position to target observation position: {np.mean(dist_from_chosen_to_target_list)}")
            logging.info(f"Wrong snapshot count: {wrong_snapshot_count}/{question_ind - success_count}")
            logging.info(f"Too many steps count: {too_many_steps_count}/{question_ind - success_count}")
            logging.info(f"Other errors count: {other_errors_count}/{question_ind - success_count}")
            logging.info(f"#######################################################\n")

            # print the items in the scene graph
            snapshot_dict = {}
            for obj_id, obj in scene.objects.items():
                if obj['image'] not in snapshot_dict:
                    snapshot_dict[obj['image']] = []
                snapshot_dict[obj['image']].append(
                    f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
                )
            logging.info(f"Scene graph of question {question_id}:")
            for snapshot_id, obj_list in snapshot_dict.items():
                logging.info(f"{snapshot_id}:")
                for obj_str in obj_list:
                    logging.info(f"\t{obj_str}")


        logging.info(f'Scene {scene_id} finish')

    result_dict = {}
    result_dict["success_rate"] = success_count / question_ind
    result_dict["same_class_ratio"] = same_class_count / question_ind
    result_dict["mean_path_length"] = np.mean(path_length_list)
    result_dict["mean_success_path_length"] = np.mean([x for i, x in enumerate(path_length_list) if success_list[i] == 1])
    result_dict["mean_distance_from_chosen_to_target"] = np.mean(dist_from_chosen_to_target_list)
    result_dict["wrong_snapshot_ratio"] = wrong_snapshot_count / (question_ind - success_count)
    result_dict["too_many_steps_ratio"] = too_many_steps_count / (question_ind - success_count)
    result_dict["other_errors_ratio"] = other_errors_count / (question_ind - success_count)
    with open(os.path.join(cfg.output_dir, "result.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

    logging.info(f'All scenes finish')


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(str(cfg.output_dir), "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
