import matplotlib.pyplot as plt
import matplotlib.image
import quaternion
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
import time
from PIL import Image

np.set_printoptions(precision=3)
import pickle
import json
import logging
import glob
import open_clip
from ultralytics import SAM, YOLOWorld
from hydra import initialize, compose
from habitat_sim.utils.common import quat_to_angle_axis
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
    # logging.info('prefiltering prompt')
    # logging.info(
    #     tokenizer.decode(filter_input_ids[0][filter_input_ids[0] != tokenizer.pad_token_id])
    # )
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
    # print("the output of prefiltering", filter_outputs)
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
    # logging.info('final input to the model')
    # logging.info(
    #     tokenizer.decode(input_ids[0][input_ids[0] != tokenizer.pad_token_id])
    # )
    # input()
    # the loss of : exists in infer_selection
    # but in final prompt
    if len(torch.where(sample.input_ids==22550)[1]) == 0:
        logging.info(f"invalid: no token 'answer'!")
        return None
        eixt(0)  # ???
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


def main(cfg, start_ratio=0.0, end_ratio=1.0):
    # use hydra to load concept graph related configs
    with initialize(config_path="conceptgraph/hydra_configs", job_name="app"):
        cfg_cg = compose(config_name=cfg.concept_graph_config_name)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    # questions_list = sorted(questions_list, key=lambda x: x['question_id'])
    # shuffle the data
    random.shuffle(questions_list)

    scene_id_to_questions = {}
    for question_data in questions_list:
        scene_id = question_data["episode_history"]

        if '00853' in scene_id:
            logging.info(f"Skip scene 00853")
            continue

        if scene_id not in scene_id_to_questions:
            scene_id_to_questions[scene_id] = []
        scene_id_to_questions[scene_id].append(question_data)
    logging.info(f"Number of scenes in total: {len(scene_id_to_questions)}")
    logging.info(f"Number of questions in total: {len(questions_list)}")

    # split the test data by scene
    scene_id_split = list(scene_id_to_questions.keys())[int(start_ratio * len(scene_id_to_questions)):int(end_ratio * len(scene_id_to_questions))]
    questions_list = [question_data for question_data in questions_list if question_data["episode_history"] in scene_id_split]
    scene_id_to_questions = {scene_id: question_data for scene_id, question_data in scene_id_to_questions.items() if scene_id in scene_id_split}
    logging.info(f"Number of scenes in split: {len(scene_id_to_questions)}")
    logging.info(f"Number of questions in split: {len(questions_list)}")

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

    # load success list and path length list
    if os.path.exists(os.path.join(str(cfg.output_dir), "success_list.pkl")):
        with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "rb") as f:
            success_list = pickle.load(f)
    else:
        success_list = []
    if os.path.exists(os.path.join(str(cfg.output_dir), "path_length_list.pkl")):
        with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "rb") as f:
            path_length_list = pickle.load(f)
    else:
        path_length_list = {}

    success_count = 0
    max_target_observation = cfg.max_target_observation
    question_idx = -1

    for scene_id, questions_in_scene in scene_id_to_questions.items():
        finished_questions = []

        while True:
            if len(finished_questions) == len(questions_in_scene):
                logging.info(f"Scene {scene_id} finished!")
                break

            logging.info(f"Loading scene {scene_id}")

            # load scene
            try:
                del scene
            except:
                pass

            scene = Scene(scene_id, cfg, cfg_cg)

            # Set the classes for the detection model
            detection_model.set_classes(scene.obj_classes.get_classes_arr())

            episode_data_dir = os.path.join(str(cfg.output_dir), f"{scene_id}_0")
            episode_data_dir_count = 0
            while os.path.exists(episode_data_dir):
                episode_data_dir_count += 1
                episode_data_dir = os.path.join(str(cfg.output_dir), f"{scene_id}_{episode_data_dir_count}")

            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)
            os.makedirs(episode_snapshot_dir, exist_ok=True)

            init_pts = questions_in_scene[0]["position"]
            init_quat = quaternion.quaternion(*questions_in_scene[0]["rotation"])

            pts = np.asarray(init_pts)
            angle, axis = quat_to_angle_axis(init_quat)
            angle = angle * axis[1] / np.abs(axis[1])
            rotation = get_quaternion(angle, 0)

            # initialize the TSDF
            pts_normal = pos_habitat_to_normal(pts)
            floor_height = pts_normal[-1]
            tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
            num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
            num_step = max(num_step, 50)
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

            logging.info(f'\n\nScene {scene_id} initialization successful!')

            # run questions in the scene
            global_step = -1
            all_snapshot_features = {}
            finished_question_count = 0

            for question_data in questions_in_scene:
                if finished_question_count == cfg.clear_up_scene_interval or len(scene.snapshots) > cfg.max_snapshot_threshold:
                    logging.info(f"Clear up the scene after {cfg.clear_up_scene_interval} questions, or {len(scene.snapshots)} snapshots")
                    break

                question_id = question_data['question_id']
                question = question_data['question']
                answer = question_data['answer']

                if question_id in finished_questions:
                    continue

                question_idx += 1

                logging.info(f"\n========\nQuestion id {question_id}")

                episode_object_observe_dir = os.path.join(str(cfg.output_dir), question_id, 'object_observations')
                os.makedirs(episode_object_observe_dir, exist_ok=True)

                if len(os.listdir(episode_object_observe_dir)) > 0:
                    logging.info(f"Question id {question_id} already has enough target observations!")
                    success_count += 1
                    finished_questions.append(question_id)
                    continue

                # init the start point
                pts = np.asarray(question_data["position"])
                angle, axis = quat_to_angle_axis(quaternion.quaternion(*question_data["rotation"]))
                angle = angle * axis[1] / np.abs(axis[1])
                rotation = get_quaternion(angle, 0)
                pts_normal = pos_habitat_to_normal(pts)

                # record the history of the agent's path
                pts_pixs = np.empty((0, 2))
                pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

                # run steps
                target_found = False
                explore_dist = 0.0
                cnt_step = -1
                target_observation_count = 0
                memory_feature = None

                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                max_point_choice = None

                while cnt_step < num_step - 1:
                    cnt_step += 1
                    global_step += 1
                    logging.info(f"\n== step: {cnt_step}, global step: {global_step} ==")
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

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        with torch.no_grad():
                            annotated_rgb, added_obj_ids, target_obj_id_det = scene.update_scene_graph(
                                image_rgb=rgb[..., :3], depth=depth, intrinsics=cam_intr, cam_pos=cam_pose,
                                detection_model=detection_model, sam_predictor=sam_predictor, clip_model=clip_model,
                                clip_preprocess=clip_preprocess, clip_tokenizer=clip_tokenizer,
                                pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
                                img_path=obs_file_name,
                                frame_idx=cnt_step * total_views + view_idx,
                                target_obj_mask=None,
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
                            else:
                                plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), rgb)
                            all_added_obj_ids += added_obj_ids

                        # clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(frame_idx=cnt_step * total_views + view_idx, pts=pts)

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

                    # Turn to face each frontier point and get rgb image
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
                                    os.path.join(episode_frontier_dir, f"{global_step}_{i}.png"),
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
                            frontier.image = f"{global_step}_{i}.png"
                            frontier.feature = img_feature

                    if cfg.choose_every_step:
                        if tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier:
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

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
                            rgb_egocentric_views_features_tensor = torch.cat(rgb_egocentric_views_features, dim=0)
                            step_dict["egocentric_view_features"] = rgb_egocentric_views_features_tensor
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

                            # print the items in the scene graph
                            snapshot_dict = {}
                            for obj_id, obj in scene.objects.items():
                                if obj['image'] not in snapshot_dict:
                                    snapshot_dict[obj['image']] = []
                                snapshot_dict[obj['image']].append(
                                    f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
                                )
                            for snapshot_id, obj_list in snapshot_dict.items():
                                logging.info(f"{snapshot_id}:")
                                for obj_str in obj_list:
                                    logging.info(f"\t{obj_str}")
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

                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts_normal,
                            objects=scene.objects,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                            random_position=False if target_observation_count == 0 else True  # use the best observation point for the first observation, and random for the rest
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

                        fig.tight_layout()
                        plt.savefig(os.path.join(visualization_path, f"{global_step}_{question_id}"))
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
                        plt.savefig(os.path.join(frontier_video_path, f'{global_step}_{question_id}.png'))
                        plt.close()

                    # update position and rotation
                    pts_normal = np.append(pts_normal, floor_height)
                    pts = pos_normal_to_habitat(pts_normal)
                    rotation = get_quaternion(angle, 0)
                    explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                    logging.info(f"Current position: {pts}, {explore_dist:.3f}")

                    if type(max_point_choice) == SnapShot and target_arrived:
                        # get an observation and break
                        obs, _ = scene.get_observation(pts, angle)
                        rgb = obs["color_sensor"]

                        plt.imsave(
                            os.path.join(episode_object_observe_dir, f"target_{target_observation_count}.png"), rgb
                        )
                        # also, save the snapshot image itself
                        snapshot_filename = max_point_choice.image.split(".")[0]
                        os.system(f"cp {os.path.join(episode_snapshot_dir, max_point_choice.image)} {os.path.join(episode_object_observe_dir, f'snapshot_{snapshot_filename}.png')}")

                        target_observation_count += 1
                        if target_observation_count >= max_target_observation:
                            target_found = True
                            break

                # here once the model has chosen one snapshot, we count it as a success
                if target_observation_count > 0:
                    target_found = True

                if target_found:
                    success_count += 1
                    # We only consider samples that model predicts object (use baseline results other samples for now)
                    # TODO: you can save path_length in the same format as you did for the baseline
                    if question_id not in success_list:
                        success_list.append(question_id)
                    path_length_list[question_id] = explore_dist
                    logging.info(f"Question id {question_id} finish with {cnt_step} steps, {explore_dist} length")
                else:
                    logging.info(f"Question id {question_id} failed, {explore_dist} length")
                logging.info(f"{question_idx + 1}/{total_questions}: Success rate: {success_count}/{question_idx + 1}")
                logging.info(f"Mean path length for success exploration: {np.mean(list(path_length_list.values()))}")
                # logging.info(f'Scene {scene_id} finish')

                # if target not found, select images from existing snapshots for question answering
                if not target_found:
                    if cfg.handle_target_not_found == 'none':
                        selected_snapshot_ids = []
                    elif cfg.handle_target_not_found == "random":
                        all_snapshot_ids = list(scene.snapshots.keys())
                        selected_snapshot_ids = random.sample(all_snapshot_ids, len(cfg.num_final_images))
                    elif cfg.handle_target_not_found == "with_model":
                        selected_snapshot_ids = []
                        for _ in range(cfg.num_final_images):
                            step_dict = {}
                            # add snapshot features
                            step_dict["snapshot_features"] = {}
                            step_dict["snapshot_objects"] = {}
                            snapshot_id_mapping = {}
                            prompt_ss_idx = 0
                            for snapshot_idx, (rgb_id, snapshot) in enumerate(scene.snapshots.items()):
                                if rgb_id in selected_snapshot_ids:
                                    continue
                                step_dict["snapshot_features"][rgb_id] = all_snapshot_features[rgb_id]
                                step_dict["snapshot_objects"][rgb_id] = snapshot.cluster

                                snapshot_id_mapping[prompt_ss_idx] = snapshot_idx
                                prompt_ss_idx += 1

                            if len(snapshot_id_mapping) == 0:
                                logging.info(f"Question id {question_id} target not found handling: no snapshots available!")
                                break

                            # add scene graph
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

                            # we don't need to add frontier for model selection
                            step_dict["frontiers"] = []

                            if cfg.egocentric_views:
                                step_dict["egocentric_view_features"] = rgb_egocentric_views_features_tensor
                                step_dict["use_egocentric_views"] = True

                            if cfg.action_memory:
                                step_dict["memory_feature"] = memory_feature
                                step_dict["use_action_memory"] = True

                            step_dict["frontier_features"] = None
                            step_dict["frontier_positions"] = None
                            step_dict["question"] = question
                            step_dict["scene"] = scene_id
                            if cfg.prefiltering:
                                outputs, _ = inference(model, tokenizer, step_dict, cfg)
                            else:
                                outputs = inference(model, tokenizer, step_dict, cfg)
                            if outputs is None:
                                logging.info(f"Question id {question_id} target not found handling: model generation error!")
                                continue
                            try:
                                target_type, target_index = outputs.split(" ")[0], outputs.split(" ")[1]
                                logging.info(f"Prediction in target not found handling: {target_type}, {target_index}")
                            except:
                                logging.info(f"Wrong output format in target not found handling, failed!")
                                continue

                            if target_type != "snapshot":
                                logging.info(f"Invalid prediction type in target not found handling: {target_type}, failed!")
                                continue

                            if int(target_index) < 0 or int(target_index) >= len(snapshot_id_mapping):
                                logging.info(f"target index can not match real objects in target not found handling: {target_index}, failed!")
                                continue
                            target_index = snapshot_id_mapping[int(target_index)]
                            pred_target_snapshot = list(scene.snapshots.values())[int(target_index)]

                            assert pred_target_snapshot.image not in selected_snapshot_ids, f"{pred_target_snapshot.image} already selected"
                            selected_snapshot_ids.append(pred_target_snapshot.image)
                            logging.info(f"Handling target not found: choose snapshot {pred_target_snapshot.image}")

                    # save the selected images
                    for ss_id in selected_snapshot_ids:
                        os.system(f"cp {os.path.join(episode_snapshot_dir, ss_id)} {os.path.join(episode_object_observe_dir, f'snapshot_{ss_id}')}")



                # print the items in the scene graph
                snapshot_dict = {}
                for obj_id, obj in scene.objects.items():
                    if obj['image'] not in snapshot_dict:
                        snapshot_dict[obj['image']] = []
                    snapshot_dict[obj['image']].append(
                        f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
                    )
                logging.info(f"Scene graph of question {question_id}:")
                logging.info(f"Question: {question}")
                logging.info(f"Answer: {answer}")
                for snapshot_id, obj_list in snapshot_dict.items():
                    logging.info(f"{snapshot_id}:")
                    for obj_str in obj_list:
                        logging.info(f"\t{obj_str}")

                with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
                    pickle.dump(success_list, f)
                with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
                    pickle.dump(path_length_list, f)

                finished_questions.append(question_id)
                finished_question_count += 1

    with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)

    logging.info(f'All scenes finish')


if __name__ == "__main__":
    import argparse
    from omegaconf import OmegaConf

    # get config path
    parser = argparse.ArgumentParser()
    parser.add_argument("-cf", "--cfg_file", help="cfg file path", default="", type=str)
    parser.add_argument("--start_ratio", help="start ratio", default=0.0, type=float)
    parser.add_argument("--end_ratio", help="end ratio", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    logging_path = os.path.join(str(cfg.output_dir), "log.log")


    class ElapsedTimeFormatter(logging.Formatter):
        def __init__(self, fmt=None, datefmt=None):
            super().__init__(fmt, datefmt)
            self.start_time = time.time()

        def formatTime(self, record, datefmt=None):
            elapsed_seconds = record.created - self.start_time
            hours, remainder = divmod(elapsed_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"


    # Set up the logging configuration
    formatter = ElapsedTimeFormatter(fmt="%(asctime)s - %(message)s")

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    # Set the custom formatter
    for handler in logging.getLogger().handlers:
        handler.setFormatter(formatter)

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg, start_ratio=args.start_ratio, end_ratio=args.end_ratio)
