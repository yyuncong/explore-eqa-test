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
from collections import defaultdict

np.set_printoptions(precision=3)
import pickle
import json
import logging
import glob
import open_clip
from ultralytics import SAM, YOLOWorld
from hydra import initialize, compose
from habitat_sim.utils.common import quat_to_angle_axis, quat_from_coeffs
import habitat_sim
from src.habitat import (
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
)
from src.geom import get_cam_intr, get_scene_bnds
from src.tsdf_new_cg import TSDFPlanner, Frontier, SnapShot
from src.scene_goatbench import Scene
from src.eval_utils_goatbench import rgba2rgb
from src.eval_utils_gpt import explore_step


def resize_image(image, target_h, target_w):
    # image: np.array, h, w, c
    image = Image.fromarray(image)
    image = image.resize((target_w, target_h))
    return np.array(image)


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
    scene_data_list = os.listdir(cfg.test_data_dir)
    num_scene = len(scene_data_list)
    random.shuffle(scene_data_list)

    # split the test data by scene
    # scene_data_list = scene_data_list[int(start_ratio * num_scene):int(end_ratio * num_scene)]
    num_episode = 0
    for scene_data_file in scene_data_list:
        with open(os.path.join(cfg.test_data_dir, scene_data_file), 'r') as f:
            num_episode += int(len(json.load(f)['episodes']) * (end_ratio - start_ratio))
    logging.info(f"Total number of episodes: {num_episode}")
    logging.info(f"Total number of scenes: {len(scene_data_list)}")

    all_scene_ids = os.listdir(cfg.scene_data_path_train + '/train') + os.listdir(cfg.scene_data_path_val + '/val')

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

    # load result stats
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_by_snapshot_{start_ratio}_{end_ratio}.pkl")):
        success_by_snapshot = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"success_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        success_by_snapshot = {}  # subtask_id -> success
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl")):
        success_by_distance = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        success_by_distance = {}  # subtask id -> success
    if os.path.exists(os.path.join(str(cfg.output_dir), f"spl_by_snapshot_{start_ratio}_{end_ratio}.pkl")):
        spl_by_snapshot = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"spl_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        spl_by_snapshot = {}  # subtask id -> spl
    if os.path.exists(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl")):
        spl_by_distance = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        spl_by_distance = {}  # subtask id -> spl
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl")):
        success_by_task = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        #success_by_task = {}  # task type -> success
        success_by_task = defaultdict(list)
    if os.path.exists(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl")):
        spl_by_task = pickle.load(
            open(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl"), "rb")
        )
    else:
        #spl_by_task = {}  # task type -> spl
        spl_by_task = defaultdict(list)
    assert len(success_by_snapshot) == len(spl_by_snapshot) == len(success_by_distance) == len(spl_by_distance), f"{len(success_by_snapshot)} != {len(spl_by_snapshot)} != {len(success_by_distance)} != {len(spl_by_distance)}"
    assert sum([len(task_res) for task_res in success_by_task.values()]) == sum([len(task_res) for task_res in spl_by_task.values()]) == len(success_by_snapshot), f"{sum([len(task_res) for task_res in success_by_task.values()])} != {sum([len(task_res) for task_res in spl_by_task.values()])} != {len(success_by_snapshot)}"

    question_idx = -1
    for scene_data_file in scene_data_list:
        scene_name = scene_data_file.split(".")[0]
        scene_id = [scene_id for scene_id in all_scene_ids if scene_name in scene_id][0]
        scene_data = json.load(open(os.path.join(cfg.test_data_dir, scene_data_file), "r"))
        total_episodes = len(scene_data["episodes"])

        navigation_goals = scene_data["goals"]  # obj_id to obj_data, apply for all episodes in this scene

        for episode_idx, episode in enumerate(scene_data["episodes"][int(start_ratio * total_episodes):int(end_ratio * total_episodes)]):
            logging.info(f"Episode {episode_idx + 1}/{total_episodes}")
            logging.info(f"Loading scene {scene_id}")
            episode_id = episode["episode_id"]

            # filter the task according to goatbench
            filtered_tasks = []
            for goal in episode["tasks"]:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                dset_same_cat_goals = [
                    x
                    for x in navigation_goals.values()
                    if x[0]["object_category"] == goal_category
                ]

                if goal_type == "description":
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    if len(goal_inst[0]["lang_desc"].split(" ")) <= 55:
                        filtered_tasks.append(goal)
                else:
                    filtered_tasks.append(goal)

            all_subtask_goals = []
            all_subtask_goal_types = []
            for goal in filtered_tasks:
                goal_type = goal[1]
                goal_category = goal[0]
                goal_inst_id = goal[2]

                all_subtask_goal_types.append(goal_type)

                dset_same_cat_goals = [
                    x
                    for x in navigation_goals.values()
                    if x[0]["object_category"] == goal_category
                ]
                children_categories = dset_same_cat_goals[0][0][
                    "children_object_categories"
                ]
                for child_category in children_categories:
                    goal_key = f"{scene_name}.basis.glb_{child_category}"
                    if goal_key not in navigation_goals:
                        print(f"!!! {goal_key} not in navigation_goals")
                        continue
                    print(f"!!! {goal_key} added")
                    dset_same_cat_goals[0].extend(navigation_goals[goal_key])

                assert (
                        len(dset_same_cat_goals) == 1
                ), f"more than 1 goal categories for {goal_category}"

                if goal_type == "object":
                    all_subtask_goals.append(dset_same_cat_goals[0])
                else:
                    goal_inst = [
                        x
                        for x in dset_same_cat_goals[0]
                        if x["object_id"] == goal_inst_id
                    ]
                    all_subtask_goals.append(goal_inst)

            # check whether this episode has been processed
            finished_subtask_ids = list(success_by_snapshot.keys())
            finished_episode_subtask = [subtask_id for subtask_id in finished_subtask_ids if subtask_id.startswith(f"{scene_id}_{episode_id}_")]
            if len(finished_episode_subtask) >= len(all_subtask_goals):
                logging.info(f"Scene {scene_id} Episode {episode_id} already done!")
                continue

            # load scene
            try:
                del scene
            except:
                pass

            scene = Scene(scene_id, cfg, cfg_cg)

            # Set the classes for the detection model
            detection_model.set_classes(scene.obj_classes.get_classes_arr())

            episode_data_dir = os.path.join(str(cfg.output_dir), f"{scene_id}_ep_{episode_id}")
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)
            os.makedirs(episode_snapshot_dir, exist_ok=True)

            init_pts = episode["start_position"]
            init_quat = quat_from_coeffs(episode["start_rotation"])

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
            all_snapshots = {}
            for subtask_idx, (goal_type, subtask_goal) in enumerate(zip(all_subtask_goal_types, all_subtask_goals)):
                subtask_id = f"{scene_id}_{episode_id}_{subtask_idx}"

                # determine the navigation goals
                goal_category = subtask_goal[0]["object_category"]
                goal_obj_ids = [x["object_id"] for x in subtask_goal]
                goal_obj_ids = [int(x.split('_')[-1]) for x in goal_obj_ids]
                if goal_type != "object":
                    assert len(goal_obj_ids) == 1, f"{len(goal_obj_ids)} != 1"

                goal_positions = [x["position"] for x in subtask_goal]
                goal_positions_voxel = [tsdf_planner.world2vox(pos_habitat_to_normal(p)) for p in goal_positions]
                goal_obj_ids_mapping = {obj_id: [] for obj_id in goal_obj_ids}

                viewpoints = [
                    view_point["agent_state"]["position"] for goal in subtask_goal for view_point in goal["view_points"]
                ]
                # get the shortest distance from current position to the viewpoints
                all_distances = []
                for viewpoint in viewpoints:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = pts
                    path.requested_end = viewpoint
                    found_path = scene.pathfinder.find_path(path)
                    if not found_path:
                        all_distances.append(np.inf)
                    else:
                        all_distances.append(path.geodesic_distance)
                start_end_subtask_distance = min(all_distances)

                logging.info(f"\nScene {scene_id} Episode {episode_id} Subtask {subtask_idx + 1}/{len(all_subtask_goals)}")

                subtask_object_observe_dir = os.path.join(str(cfg.output_dir), f"{subtask_id}", 'object_observations')
                if os.path.exists(subtask_object_observe_dir):
                    os.system(f"rm -r {subtask_object_observe_dir}")
                os.makedirs(subtask_object_observe_dir, exist_ok=False)

                # Prepare metadata for the subtask
                subtask_metadata = {
                    "question_id": subtask_id,
                    "episode_history": scene_id,
                    "category": "object localization",
                    "question": None,
                    "image": None,
                    "answer": goal_category,
                    "object_id": goal_obj_ids,  # this is a list of obj id, since for object class type, there will be multiple target objects
                    "class": goal_category,
                    "position": goal_positions, # also a list of positions for possible multiple objects
                    "task_type": goal_type
                }
                # format question according to the goal type
                if goal_type == "object":
                    subtask_metadata['question'] = f"Where is the {goal_category}?"
                elif goal_type == "description":
                    subtask_metadata['question'] = f"Could you find the object described as \'{subtask_goal[0]['lang_desc']}\'?"
                else:  # goal_type == "image"
                    subtask_metadata['question'] = f"Could you find the object captured in the following image?"
                    view_pos_dict = random.choice(subtask_goal[0]["view_points"])['agent_state']
                    obs,_ = scene.get_observation(pts=view_pos_dict["position"], rotation=view_pos_dict["rotation"])
                    plt.imsave(os.path.join(str(cfg.output_dir), f"{subtask_id}", "image_goal.png"), obs["color_sensor"])
                    subtask_metadata["image"] = f"{cfg.output_dir}/{subtask_id}/image_goal.png"
                
                # record the history of the agent's path
                pts_pixs = np.empty((0, 2))
                pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

                # run steps
                target_found = False
                subtask_explore_dist = 0.0
                cnt_step = -1
                question_idx += 1

                # reset tsdf planner
                tsdf_planner.max_point = None
                tsdf_planner.target_point = None
                max_point_choice = None

                if cfg.clear_up_memory_every_subtask and subtask_idx > 0:
                    scene.clear_up_detections()
                    tsdf_planner = TSDFPlanner(
                        vol_bnds=tsdf_bnds,
                        voxel_size=cfg.tsdf_grid_size,
                        floor_height_offset=0,
                        pts_init=pts_normal,
                        init_clearance=cfg.init_clearance * 2,
                    )
                    all_snapshots = {}

                while cnt_step < num_step - 1:
                    cnt_step += 1
                    global_step += 1
                    logging.info(f"\n== step: {cnt_step}, global step: {global_step} ==")
                    step_dict = {}
                    if cnt_step == 0:
                        angle_increment = cfg.extra_view_angle_deg_phase_2 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_2
                    else:
                        angle_increment = cfg.extra_view_angle_deg_phase_1 * np.pi / 180
                        total_views = 1 + cfg.extra_view_phase_1
                    all_angles = [angle + angle_increment * (i - total_views // 2) for i in range(total_views)]
                    # let the main viewing angle be the last one to avoid potential overwriting problems
                    main_angle = all_angles.pop(total_views // 2)
                    all_angles.append(main_angle)

                    # observe and update the TSDF
                    rgb_egocentric_views = []
                    all_added_obj_ids = []
                    for view_idx, ang in enumerate(all_angles):
                        obs, cam_pose = scene.get_observation(pts, angle=ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]
                        rgb = rgba2rgb(rgb)

                        cam_pose_normal = pose_habitat_to_normal(cam_pose)
                        cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                        # collect all view features
                        obs_file_name = f"{global_step}-view_{view_idx}.png"
                        with torch.no_grad():
                            annotated_rgb, added_obj_ids, target_obj_id_mapping = scene.update_scene_graph(
                                image_rgb=rgb[..., :3], depth=depth, intrinsics=cam_intr, cam_pos=cam_pose,
                                detection_model=detection_model, sam_predictor=sam_predictor, clip_model=clip_model,
                                clip_preprocess=clip_preprocess, clip_tokenizer=clip_tokenizer,
                                pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
                                img_path=obs_file_name,
                                frame_idx=cnt_step * total_views + view_idx,
                                semantic_obs=semantic_obs,
                                gt_target_obj_ids=goal_obj_ids,
                            )
                            resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                            all_snapshots[obs_file_name] = resized_rgb
                            rgb_egocentric_views.append(resized_rgb)
                            if cfg.save_visualization or cfg.save_frontier_video:
                                plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), annotated_rgb)
                            else:
                                plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), rgb)
                            # update the mapping of hm3d object id to our detected object id
                            for gt_goal_id, det_goal_id in target_obj_id_mapping.items():
                                goal_obj_ids_mapping[gt_goal_id].append(det_goal_id)
                            all_added_obj_ids += added_obj_ids

                        # clean up or merge redundant objects periodically
                        scene.periodic_cleanup_objects(
                            frame_idx=cnt_step * total_views + view_idx, pts=pts,
                            goal_obj_ids_mapping=goal_obj_ids_mapping
                        )

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
                    scene.update_snapshots(obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)
                    logging.info(f"Step {cnt_step} {len(scene.objects)} objects, {len(scene.snapshots)} snapshots")

                    # update the mapping of object id to class name, since the objects have been updated
                    object_id_to_name = {obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()}
                    step_dict["obj_map"] = object_id_to_name

                    step_dict["snapshot_objects"] = {}
                    step_dict["snapshot_imgs"] = {}
                    for rgb_id, snapshot in scene.snapshots.items():
                        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
                        step_dict["snapshot_imgs"][rgb_id] = all_snapshots[rgb_id]

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
                            processed_rgb = resize_image(rgba2rgb(frontier_obs), cfg.prompt_h, cfg.prompt_w)
                            frontier.image = f"{global_step}_{i}.png"
                            frontier.feature = processed_rgb

                    if cfg.choose_every_step:
                        if tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier:
                            # reset target point to allow the model to choose again
                            tsdf_planner.max_point = None
                            tsdf_planner.target_point = None

                    logging.info(f"Goal object mapping: {goal_obj_ids_mapping}")

                    # use the most common id in the mapped ids as the detected target object id
                    target_obj_ids_estimate = []
                    for obj_id, det_ids in goal_obj_ids_mapping.items():
                        if len(det_ids) == 0:
                            continue
                        target_obj_ids_estimate.append(max(set(det_ids), key=det_ids.count))

                    if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                        # choose a frontier, and set it as the explore target
                        step_dict["frontiers"] = []
                        # since we skip the stuck frontier for input of the vlm, we need to map the
                        # vlm output frontier id to the tsdf planner frontier id
                        ft_id_to_vlm_id = {}
                        vlm_id_count = 0
                        for i, frontier in enumerate(tsdf_planner.frontiers):
                            frontier_dict = {}
                            assert frontier.image is not None and frontier.feature is not None
                            frontier_dict["rgb_id"] = frontier.image
                            frontier_dict["img"] = frontier.feature

                            step_dict["frontiers"].append(frontier_dict)

                            ft_id_to_vlm_id[i] = vlm_id_count
                            vlm_id_count += 1
                        vlm_id_to_ft_id = {v: k for k, v in ft_id_to_vlm_id.items()}

                        if cfg.egocentric_views:
                            step_dict["egocentric_views"] = rgb_egocentric_views
                            step_dict["use_egocentric_views"] = True

                        # add model prediction here
                        if len(step_dict["frontiers"]) > 0:
                            step_dict["frontier_imgs"] = [frontier["img"] for frontier in step_dict["frontiers"]]
                        else:
                            step_dict["frontier_imgs"] = []
                        step_dict["question"] = subtask_metadata["question"]#question
                        step_dict["scene"] = scene_id
                        step_dict["task_type"] = subtask_metadata["task_type"]
                        step_dict["class"] = subtask_metadata["class"]
                        step_dict["image"] = subtask_metadata["image"]

                        outputs, snapshot_id_mapping = explore_step(step_dict, cfg)
                        if outputs is None:
                            # encounter generation error
                            logging.info(f"Subtask id {subtask_id} invalid: model generation error!")
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

                        if max_point_choice is None:
                            logging.info(f"Subtask id {subtask_id} invalid: no valid choice!")
                            break

                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts_normal,
                            objects=scene.objects,
                            cfg=cfg.planner,
                            pathfinder=scene.pathfinder,
                        )
                        if not update_success:
                            logging.info(f"Subtask id {subtask_id} invalid: set_next_navigation_point failed!")
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
                        logging.info(f"Subtask id {subtask_id} invalid: agent_step failed!")
                        break
                    pts_normal, angle, pts_pix, fig, _, target_arrived = return_values

                    # sanity check
                    obj_exclude_count = sum([1 if obj['num_detections'] < cfg.min_detection else 0 for obj in scene.objects.values()])
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
                        if scene.objects[obj_id]['num_detections'] < cfg.min_detection:
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
                        for goal_id, goal_pos_voxel in zip(goal_obj_ids, goal_positions_voxel):
                            color = 'green' if len(goal_obj_ids_mapping[goal_id]) > 0 else 'red'
                            ax5.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)
                            ax1, ax2, ax4 = fig.axes[0], fig.axes[1], fig.axes[3]
                            ax4.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)
                            ax1.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)
                            ax2.scatter(goal_pos_voxel[1], goal_pos_voxel[0], c=color, s=120)

                        fig.tight_layout()
                        plt.savefig(os.path.join(visualization_path, f"{global_step}_{subtask_id}"))
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
                        global_caption = f"{subtask_metadata['question']}\n{subtask_metadata['task_type']}\n{subtask_metadata['class']}"
                        fig.suptitle(global_caption, fontsize=16)
                        plt.tight_layout(rect=(0., 0., 1., 0.95))
                        plt.savefig(os.path.join(frontier_video_path, f'{global_step}_{subtask_id}.png'))
                        plt.close()

                    # update position and rotation
                    pts_normal = np.append(pts_normal, floor_height)
                    pts = pos_normal_to_habitat(pts_normal)
                    rotation = get_quaternion(angle, 0)
                    subtask_explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                    logging.info(f"Current position: {pts}, {subtask_explore_dist:.3f}")

                    if type(max_point_choice) == SnapShot and target_arrived:
                        # get an observation and break
                        obs, _ = scene.get_observation(pts, angle=angle)
                        rgb = obs["color_sensor"]

                        plt.imsave(os.path.join(subtask_object_observe_dir, f"target.png"), rgb)
                        # also, save the snapshot image itself
                        snapshot_filename = max_point_choice.image.split(".")[0]
                        os.system(f"cp {os.path.join(episode_snapshot_dir, max_point_choice.image)} {os.path.join(subtask_object_observe_dir, f'snapshot_{snapshot_filename}.png')}")

                        target_found = True
                        break

                # get some statistics
                # check whether the target objects are in the selected snapshot
                if target_found and np.any([obj_id in max_point_choice.cluster for obj_id in target_obj_ids_estimate]):
                    success_by_snapshot[subtask_id] = 1.0
                    logging.info(f"Success: {target_obj_ids_estimate} in chosen snapshot {max_point_choice.image}!")
                else:
                    success_by_snapshot[subtask_id] = 0.0
                    logging.info(f"Fail: {target_obj_ids_estimate} not in chosen snapshot {max_point_choice.image}!")

                # calculate the distance to the nearest view point
                all_distances = []
                for viewpoint in viewpoints:
                    path = habitat_sim.ShortestPath()
                    path.requested_start = pts
                    path.requested_end = viewpoint
                    found_path = scene.pathfinder.find_path(path)
                    if not found_path:
                        all_distances.append(np.inf)
                    else:
                        all_distances.append(path.geodesic_distance)
                agent_subtask_distance = min(all_distances)
                if agent_subtask_distance < cfg.success_distance:
                    success_by_distance[subtask_id] = 1.0
                    logging.info(f"Success: agent reached the target viewpoint at distance {agent_subtask_distance}!")
                else:
                    success_by_distance[subtask_id] = 0.0
                    logging.info(f"Fail: agent failed to reach the target viewpoint at distance {agent_subtask_distance}!")

                # calculate the spl
                spl_by_snapshot[subtask_id] = (success_by_snapshot[subtask_id] * start_end_subtask_distance /
                                               max(start_end_subtask_distance, subtask_explore_dist))
                spl_by_distance[subtask_id] = (success_by_distance[subtask_id] * start_end_subtask_distance /
                                               max(start_end_subtask_distance, subtask_explore_dist))

                success_by_task[goal_type].append(success_by_snapshot[subtask_id])
                spl_by_task[goal_type].append(spl_by_snapshot[subtask_id])

                logging.info(f"Subtask {subtask_id} finished with {cnt_step} steps, {subtask_explore_dist} length")
                logging.info(f"Subtask spl by snapshot: {spl_by_snapshot[subtask_id]}, spl by distance: {spl_by_distance[subtask_id]}")

                logging.info(f"Success rate by snapshot: {100 * np.mean(np.asarray(list(success_by_snapshot.values()))):.2f}")
                logging.info(f"Success rate by distance: {100 * np.mean(np.asarray(list(success_by_distance.values()))):.2f}")
                logging.info(f"SPL by snapshot: {100 * np.mean(np.asarray(list(spl_by_snapshot.values()))):.2f}")
                logging.info(f"SPL by distance: {100 * np.mean(np.asarray(list(spl_by_distance.values()))):.2f}")

                for task_name, success_list in success_by_task.items():
                    logging.info(f"Success rate for {task_name}: {100 * np.mean(np.asarray(success_list)):.2f}")
                for task_name, spl_list in spl_by_task.items():
                    logging.info(f"SPL for {task_name}: {100 * np.mean(np.asarray(spl_list)):.2f}")

                # print the items in the scene graph
                snapshot_dict = {}
                for obj_id, obj in scene.objects.items():
                    if obj['image'] not in snapshot_dict:
                        snapshot_dict[obj['image']] = []
                    snapshot_dict[obj['image']].append(
                        f"{obj_id}: {obj['class_name']} {obj['num_detections']}"
                    )
                logging.info(f"Scene graph of question {subtask_id}:")
                logging.info(f"Question: {subtask_metadata['question']}")
                logging.info(f"Task type: {subtask_metadata['task_type']}")
                logging.info(f"Answer: {subtask_metadata['class']}")
                for snapshot_id, obj_list in snapshot_dict.items():
                    logging.info(f"{snapshot_id}:")
                    for obj_str in obj_list:
                        logging.info(f"\t{obj_str}")

            # save the results at the end of each episode
            assert len(success_by_snapshot) == len(spl_by_snapshot) == len(success_by_distance) == len(
                spl_by_distance), f"{len(success_by_snapshot)} != {len(spl_by_snapshot)} != {len(success_by_distance)} != {len(spl_by_distance)}"
            assert sum([len(task_res) for task_res in success_by_task.values()]) == sum(
                [len(task_res) for task_res in spl_by_task.values()]) == len(
                success_by_snapshot), f"{sum([len(task_res) for task_res in success_by_task.values()])} != {sum([len(task_res) for task_res in spl_by_task.values()])} != {len(success_by_snapshot)}"
            with open(os.path.join(str(cfg.output_dir), f"success_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(success_by_snapshot, f)
            with open(os.path.join(str(cfg.output_dir), f"spl_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(spl_by_snapshot, f)
            with open(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(success_by_distance, f)
            with open(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(spl_by_distance, f)
            with open(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(success_by_task, f)
            with open(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
                pickle.dump(spl_by_task, f)


    # save the results
    assert len(success_by_snapshot) == len(spl_by_snapshot) == len(success_by_distance) == len(
        spl_by_distance), f"{len(success_by_snapshot)} != {len(spl_by_snapshot)} != {len(success_by_distance)} != {len(spl_by_distance)}"
    assert sum([len(task_res) for task_res in success_by_task.values()]) == sum(
        [len(task_res) for task_res in spl_by_task.values()]) == len(
        success_by_snapshot), f"{sum([len(task_res) for task_res in success_by_task.values()])} != {sum([len(task_res) for task_res in spl_by_task.values()])} != {len(success_by_snapshot)}"
    with open(os.path.join(str(cfg.output_dir), f"success_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(success_by_snapshot, f)
    with open(os.path.join(str(cfg.output_dir), f"spl_by_snapshot_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(spl_by_snapshot, f)
    with open(os.path.join(str(cfg.output_dir), f"success_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(success_by_distance, f)
    with open(os.path.join(str(cfg.output_dir), f"spl_by_distance_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(spl_by_distance, f)
    with open(os.path.join(str(cfg.output_dir), f"success_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(success_by_task, f)
    with open(os.path.join(str(cfg.output_dir), f"spl_by_task_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(spl_by_task, f)

    logging.info(f'All scenes finish')

    # aggregate the results into a single file
    filenames_to_merge = ['success_by_snapshot', 'spl_by_snapshot', 'success_by_distance', 'spl_by_distance']
    for filename in filenames_to_merge:
        all_results = {}
        all_results_paths = glob.glob(os.path.join(str(cfg.output_dir), f"{filename}_*.pkl"))
        for results_path in all_results_paths:
            with open(results_path, "rb") as f:
                all_results.update(pickle.load(f))
        logging.info(f"Total {filename} results: {100 * np.mean(list(all_results.values())):.2f}")
        with open(os.path.join(str(cfg.output_dir), f"{filename}.pkl"), "wb") as f:
            pickle.dump(all_results, f)
    filenames_to_merge = ['success_by_task', 'spl_by_task']
    for filename in filenames_to_merge:
        all_results = {}
        all_results_paths = glob.glob(os.path.join(str(cfg.output_dir), f"{filename}_*.pkl"))
        for results_path in all_results_paths:
            with open(results_path, "rb") as f:
                separate_stat = pickle.load(f)
                for task_name, task_res in separate_stat.items():
                    if task_name not in all_results:
                        all_results[task_name] = []
                    all_results[task_name] += task_res
        for task_name, task_res in all_results.items():
            logging.info(f"Total {filename} results for {task_name}: {100 * np.mean(task_res):.2f}")
        with open(os.path.join(str(cfg.output_dir), f"{filename}.pkl"), "wb") as f:
            pickle.dump(all_results, f)


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
    logging_path = os.path.join(str(cfg.output_dir), f"log_{args.start_ratio:.2f}_{args.end_ratio:.2f}.log")

    os.system(f"cp {args.cfg_file} {cfg.output_dir}")

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
