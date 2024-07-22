import os
import random

os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # disable warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HABITAT_SIM_LOG"] = (
    "quiet"  # https://aihabitat.org/docs/habitat-sim/logging.html
)
os.environ["MAGNUM_LOG"] = "quiet"
import numpy as np

np.set_printoptions(precision=3)
import csv
import pickle
import json
import logging
import glob
import math
import torch
import quaternion
import matplotlib.pyplot as plt
import matplotlib.image
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors, quat_to_angle_axis
from src.habitat import (
    make_semantic_cfg_new,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
    get_frontier_observation
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance
from src.tsdf_new import TSDFPlanner, Frontier, SnapShot
from src.eval_utils_snapshot_new import (
    prepare_step_dict, 
    get_item, 
    encode, 
    load_scene_features, 
    rgba2rgb, load_checkpoint, 
    collate_wrapper, 
    construct_selection_prompt,
    merge_patches,
)
from src.eval_utils_snapshot_new import SCENE_TOKEN
from inference.models import YOLOWorld

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
            cfg.top_k_categories
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
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model = YOLOWorld(model_id=cfg.detection_model_name)

    # Load dataset
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    # questions_list = sorted(questions_list, key=lambda x: x['question_id'])
    # shuffle the data
    random.shuffle(questions_list)
    print("number of questions: ", total_questions)
    print("question path: ", cfg.questions_list_path)

    print("load model")
    # Initialize LLaVA model
    # model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/work/pi_chuangg_umass_edu/yuncong/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, None, model_name, device_map=None, add_multisensory_token=True
    )
    # model = model.to("cuda")
    load_checkpoint(model, cfg.model_path)
    model = model.to("cuda")
    # model = None
    model.eval()
    print("finish loading model")

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

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data['question_id']
        question = question_data['question']
        answer = question_data['answer']

        # Extract question
        scene_id = question_data["episode_history"]
        init_pts = question_data["position"]
        init_quat = quaternion.quaternion(*question_data["rotation"])
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        # load scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        assert os.path.exists(scene_mesh_path) and os.path.exists(navmesh_path), f'{scene_mesh_path}, {navmesh_path}'
        assert os.path.exists(semantic_texture_path) and os.path.exists(scene_semantic_annotation_path), f'{semantic_texture_path}, {scene_semantic_annotation_path}'

        try:
            del tsdf_planner
        except:
            pass
        try:
            simulator.close()
        except:
            pass

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        sim_cfg = make_semantic_cfg_new(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(cfg.seed)
        pathfinder.load_nav_mesh(navmesh_path)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        logging.info(f"Load scene {scene_id} successfully")

        bbox_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
        if not os.path.exists(bbox_path):
            logging.info(f"Question id {scene_id} invalid: no bbox data!")
            continue
        bounding_box_data = json.load(open(bbox_path, "r"))
        object_id_to_bbox = {int(item['id']): {'bbox': item['bbox'], 'class': item['class_name']} for item in bounding_box_data}
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

        episode_data_dir = os.path.join(str(cfg.output_dir), str(question_id))
        episode_observations_dir = os.path.join(episode_data_dir, 'observations')
        episode_object_observe_dir = os.path.join(episode_data_dir, 'object_observations')
        episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
        episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_observations_dir, exist_ok=True)
        os.makedirs(episode_object_observe_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)
        os.makedirs(episode_snapshot_dir, exist_ok=True)

        if len(os.listdir(episode_observations_dir)) >= 50:
            logging.info(f"Question id {question_id} already has enough target observations!")
            success_count += 1
            continue

        pts = init_pts
        angle, axis = quat_to_angle_axis(init_quat)
        angle = angle * axis[1] / np.abs(axis[1])
        rotation = get_quaternion(angle, 0)

        # initialize the TSDF
        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
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

        # record the history of the agent's path
        pts_pixs = np.empty((0, 2))
        pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

        logging.info(f'\n\nQuestion id {question_id} initialization successful!')

        # run steps
        target_found = False
        explore_dist = 0.0
        cnt_step = -1
        target_observation_count = 0
        memory_feature = None

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
            rgb_egocentric_views = []
            for view_idx, ang in enumerate(all_angles):
                agent_state.position = pts
                agent_state.rotation = get_quaternion(ang, 0)
                agent.set_state(agent_state)
                pts_normal = pos_habitat_to_normal(pts)

                # Update camera info
                sensor = agent.get_state().sensor_states["depth_sensor"]
                quaternion_0 = sensor.rotation
                translation_0 = sensor.position
                cam_pose = np.eye(4)
                cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
                cam_pose[:3, 3] = translation_0
                cam_pose_normal = pose_habitat_to_normal(cam_pose)
                cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                # Get observation at current pose - skip black image, meaning robot is outside the floor
                obs = simulator.get_sensor_observations()
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]
                semantic_obs = obs["semantic_sensor"]
                if cfg.save_visualization:
                    plt.imsave(
                        os.path.join(episode_observations_dir, "{}.png".format(cnt_step)), rgb
                    )
                rgb = rgba2rgb(rgb)
                rgb_egocentric_views.append(rgb)

                # collect all view features
                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                with torch.no_grad():
                    object_added, annotated_rgb = tsdf_planner.update_scene_graph(
                        detection_model=detection_model,
                        rgb=rgb[..., :3],
                        semantic_obs=semantic_obs,
                        obj_id_to_name=object_id_to_name,
                        obj_id_to_bbox=object_id_to_bbox,
                        cfg=cfg.scene_graph,
                        file_name=obs_file_name,
                        obs_point=pts,
                        return_annotated=True
                    )
                if object_added:
                    with torch.no_grad():
                        img_feature = encode(model, image_processor, rgb).mean(0)
                    img_feature = merge_patches(
                        img_feature.view(cfg.visual_feature_size, cfg.visual_feature_size, -1), 
                        cfg.patch_size
                    )
                    all_snapshot_features[obs_file_name] = img_feature.to("cpu")
                    if cfg.save_visualization:
                        plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), rgb)

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

            tsdf_planner.update_snapshots(min_num_obj_threshold=cfg.min_num_obj_threshold)
            logging.info(f'length updated scene graph {len(tsdf_planner.simple_scene_graph.keys())}')
            logging.info(f'length updated snapshots {len(tsdf_planner.snapshots.keys())}')

            step_dict["snapshot_features"] = {}
            step_dict["snapshot_objects"] = {}
            for rgb_id, snapshot in tsdf_planner.snapshots.items():
                step_dict["snapshot_features"][rgb_id] = all_snapshot_features[rgb_id]
                step_dict["snapshot_objects"][rgb_id] = snapshot.selected_obj_list
            # print(step_dict["snapshot_objects"])
            # input()

            # record current scene graph
            step_dict["scene_graph"] = list(tsdf_planner.simple_scene_graph.keys())
            step_dict["scene_graph"] = [int(x) for x in step_dict["scene_graph"]]
            step_dict["obj_map"] = object_id_to_name
            step_dict["position"] = np.array(pts)[None,]
            obj_positions_map = {
                obj["id"]: 
                (np.array(obj["bbox"][1]) + np.array(obj["bbox"][0])) / 2
                for obj in bounding_box_data
            }
            obj_positions_map = {
                key: value[[0, 2, 1]] - step_dict["position"] for key, value in obj_positions_map.items()
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

                    frontier_obs = get_frontier_observation(
                        agent, simulator, cfg, tsdf_planner,
                        view_frontier_direction=view_frontier_direction,
                        init_pts=pts,
                        camera_tilt=0,
                        max_try_count=0
                    )

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
                    assert len(rgb_egocentric_views) == total_views
                    egocentric_views_features = []
                    for rgb_view in rgb_egocentric_views:
                        processed_rgb = rgba2rgb(rgb_view)
                        with torch.no_grad():
                            img_feature = encode(model, image_processor, processed_rgb).mean(0)
                        img_feature = merge_patches(
                            img_feature.view(cfg.visual_feature_size, cfg.visual_feature_size, -1), 
                            cfg.patch_size
                        )
                        egocentric_views_features.append(img_feature)
                    egocentric_views_features = torch.cat(egocentric_views_features, dim=0)
                    step_dict["egocentric_view_features"] = egocentric_views_features.to("cpu")
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
                    if int(target_index) < 0 or int(target_index) >= len(list(tsdf_planner.snapshots.values())):
                        logging.info(f"Prediction out of range: {target_index}, {len(tsdf_planner.simple_scene_graph)}, failed!")
                        break
                    pred_target_snapshot = list(tsdf_planner.snapshots.values())[int(target_index)]
                    logging.info(
                        "pred_target_class: " + str(' '.join([object_id_to_name[obj_id] for obj_id in pred_target_snapshot.selected_obj_list]))
                    )

                    logging.info(f"Next choice Snapshot")
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
                    memory_feature = tsdf_planner.frontiers[int(target_index)].feature.to("cpu")

                if max_point_choice is None:
                    logging.info(f"Question id {question_id} invalid: no valid choice!")
                    break

                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    cfg=cfg.planner,
                    pathfinder=pathfinder,
                    random_position=False if target_observation_count == 0 else True  # use the best observation point for the first observation, and random for the rest
                )
                if not update_success:
                    logging.info(f"Question id {question_id} invalid: set_next_navigation_point failed!")
                    break

            return_values = tsdf_planner.agent_step(
                pts=pts_normal,
                angle=angle,
                pathfinder=pathfinder,
                cfg=cfg.planner,
                path_points=None,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {question_id} invalid: agent_step failed!")
                break
            pts_normal, angle, pts_pix, fig, _, target_arrived = return_values

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
            explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

            if type(max_point_choice) == SnapShot and target_arrived:
                # observe the snapshot at the target point
                agent_state_obs = habitat_sim.AgentState()
                agent_state_obs.position = pts
                agent_state_obs.rotation = rotation
                agent.set_state(agent_state_obs)
                obs = simulator.get_sensor_observations()
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

        # ensure that the observation dir has at most 50 images
        all_img_paths = glob.glob(os.path.join(episode_observations_dir, "*.png"))
        if len(all_img_paths) > 50:
            selected_img_paths = random.sample(all_img_paths, 50)
            for path in all_img_paths:
                if path not in selected_img_paths:
                    os.remove(path)

        with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)

    with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)

    logging.info(f'All scenes finish')
    try:
        simulator.close()
    except:
        pass


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
