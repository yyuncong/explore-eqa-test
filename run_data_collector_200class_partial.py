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
    get_navigable_point_to_new,
    get_frontier_observation_and_detect_target
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance
#from src.tsdf_clustering import TSDFPlanner, Frontier, SnapShot
from src.tsdf_clustering_200class import TSDFPlanner, Frontier, SnapShot
from src.detection_utils import compute_recall,format_snapshot
from inference.models import YOLOWorld
from ultralytics import YOLOWorld as YOLO

'''
This code generate object features online
'''


def main(cfg):
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model_yoloworld = YOLOWorld(model_id=cfg.yolo_world_model_name)
    detection_model = YOLO(cfg.yolo_model_name)  # yolov8x-world.pt

    # load finetuned yolo classes
    class_id_to_name = json.load(open('yolo_finetune/class_id_to_class_name.json', 'r'))
    detection_model.set_classes(list(class_id_to_name.values()))

    # Load dataset
    with open(os.path.join(cfg.question_data_path, "generated_questions.json")) as f:
        questions_data = json.load(f)
    questions_data = sorted(questions_data, key=lambda x: x["episode_history"])
    questions_data = questions_data[int(args.start * len(questions_data)):int(args.end * len(questions_data))]
    all_scene_list = list(set([q["episode_history"] for q in questions_data]))
    # all_scene_list = sorted(all_scene_list, key=lambda x: int(x.split("-")[0]))
    random.shuffle(all_scene_list)
    logging.info(f"Loaded {len(questions_data)} questions.")

    success_list = []
    total_images_record = []
    top_k_images_record = {5: [], 10: [], 15: [], 20: []}
    top_k_correct_count = {5: 0, 10: 0, 15: 0, 20: 0}


    # for each scene, answer each question
    question_ind = 0
    success_count = 0
    total_questions = len(questions_data) * cfg.paths_per_question
    for scene_id in all_scene_list:
        all_questions_in_scene = [q for q in questions_data if q["episode_history"] == scene_id]

        ##########################################################
        # if '00873' not in scene_id:
        #     continue
        # if int(scene_id.split("-")[0]) >= 800:
        #     continue
        # rand_q = np.random.randint(0, len(all_questions_in_scene) - 1)
        # all_questions_in_scene = all_questions_in_scene[rand_q:rand_q+1]
        # all_questions_in_scene = [q for q in all_questions_in_scene if '00732-Z2DQddYp1fn_60_bed_table_338871' in q['question_id']]
        # if len(all_questions_in_scene) == 0:
        #     continue
        # random.shuffle(all_questions_in_scene)
        # all_questions_in_scene = all_questions_in_scene[:2]
        # all_questions_in_scene = [q for q in all_questions_in_scene if "00109" in q['question_id']]
        ##########################################################

        # load scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        if split == "train":
            cfg.scene_data_path = cfg.scene_data_path_train
        else:
            cfg.scene_data_path = cfg.scene_data_path_val
        scene_mesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        bbox_data_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
        if not os.path.exists(scene_mesh_path) or not os.path.exists(navmesh_path) or not os.path.exists(semantic_texture_path) or not os.path.exists(scene_semantic_annotation_path):
            logging.info(f"Scene {scene_id} not exists")
            continue
        if not os.path.exists(bbox_data_path):
            logging.info(f"Scene {scene_id} bbox data not exists")
            continue

        # assert os.path.exists(scene_mesh_path) and os.path.exists(navmesh_path), f'{scene_mesh_path}, {navmesh_path}'
        # assert os.path.exists(semantic_texture_path) and os.path.exists(scene_semantic_annotation_path), f'{semantic_texture_path}, {scene_semantic_annotation_path}'

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

        # load semantic object bbox data
        bounding_box_data = json.load(open(os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json"), "r"))
        object_id_to_bbox = {int(item['id']): {'bbox': item['bbox'], 'class': item['class_name']} for item in bounding_box_data}
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

        for question_data in all_questions_in_scene:
            # for each question, generate several paths, starting from different starting points
            for path_idx in range(cfg.path_id_offset, cfg.path_id_offset + cfg.paths_per_question):
                question_ind += 1
                target_obj_id = question_data['object_id']
                target_position = question_data['position']
                target_rotation = question_data['rotation']
                episode_data_dir = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_{path_idx}")
                episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
                object_feature_save_dir = os.path.join(episode_data_dir, 'object_features')

                # if the data has already generated, skip
                if os.path.exists(episode_data_dir) and not os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                    os.system(f"rm -r {episode_data_dir}")

                os.makedirs(episode_data_dir, exist_ok=True)
                os.makedirs(episode_frontier_dir, exist_ok=True)
                os.makedirs(object_feature_save_dir, exist_ok=True)

                # get the starting points of other generated paths for this object, if there exists any
                # get all the folder in the form os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_*")
                glob_path = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_*")
                all_generated_path_dirs = glob.glob(glob_path)
                prev_start_positions = []
                for path_dir in all_generated_path_dirs:
                    metadata_path = os.path.join(path_dir, "metadata.json")
                    if os.path.exists(metadata_path):
                        with open(metadata_path, "r") as f:
                            prev_start_positions.append(json.load(f)["init_pts"])
                if len(prev_start_positions) == 0:
                    prev_start_positions = None
                else:
                    prev_start_positions = np.asarray(prev_start_positions)

                # get a navigation start point
                floor_height = target_position[1]
                tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
                scene_length = (tsdf_bnds[:2, 1] - tsdf_bnds[:1, 0]).mean()
                min_dist = cfg.min_travel_dist
                pathfinder.seed(random.randint(0, 1000000))
                start_position, path_points, travel_dist = get_navigable_point_to_new(
                    target_position, pathfinder, max_search=1000, min_dist=min_dist, max_dist=min_dist + 5,
                    prev_start_positions=prev_start_positions
                )
                if start_position is None or path_points is None:
                    logging.info(f"Cannot find a navigable path to the target object in question {question_data['question_id']}-path {path_idx}!")
                    continue

                # set the initial orientation of the agent as the initial path direction
                init_orientation = path_points[1] - path_points[0]
                init_orientation[1] = 0
                angle, axis = quat_to_angle_axis(
                    quat_from_two_vectors(np.asarray([0, 0, -1]), init_orientation)
                )
                angle = angle * axis[1] / np.abs(axis[1])
                pts = start_position.copy()
                rotation = get_quaternion(angle, 0)

                # initialize the TSDF
                pts_normal = pos_habitat_to_normal(pts)
                try:
                    del tsdf_planner
                except:
                    pass
                tsdf_planner = TSDFPlanner(
                    vol_bnds=tsdf_bnds,
                    voxel_size=cfg.tsdf_grid_size,
                    floor_height_offset=0,
                    pts_init=pos_habitat_to_normal(start_position),
                    init_clearance=cfg.init_clearance * 2,
                )

                # convert path points to normal and drop y-axis for tsdf planner
                path_points = [pos_habitat_to_normal(p) for p in path_points]
                path_points = [p[:2] for p in path_points]

                # record the history of the agent's path
                pts_pixs = np.empty((0, 2))
                pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(start_position)[:2]))

                logging.info(f'\n\nQuestion id {question_data["question_id"]}-path {path_idx} initialization successful!')

                metadata = {}
                metadata["question"] = question_data["question"]
                metadata["answer"] = question_data["answer"]
                metadata["scene"] = question_data["episode_history"]
                metadata["init_pts"] = pts.tolist()
                metadata["init_angle"] = angle
                metadata["target_obj_id"] = target_obj_id
                metadata["target_obj_class"] = question_data["class"]

                # first run certain steps as initialization
                initialize_step = random.randint(cfg.min_initialize_step, cfg.max_initialize_step)

                # run steps
                target_found = False
                previous_choice_path = None
                # in initialization, the max_explore_dist is set to a large number to ensure the agent can reach the target
                max_explore_dist = travel_dist * cfg.max_step_dist_ratio + 999
                max_step = int(travel_dist * cfg.max_step_ratio) + 999

                explore_dist = 0.0
                cnt_step = -1

                state = 'initialize'  # 'initialize' or 'explore'
                logging.info(f"Question id {question_data['question_id']}-path {path_idx} start initialization for {initialize_step} steps")

                while explore_dist < max_explore_dist and cnt_step < max_step:
                    cnt_step += 1

                    if cnt_step == initialize_step and state == 'initialize':
                        # at the end of initialization, reset the path points to the target
                        path = habitat_sim.ShortestPath()
                        path.requested_start = pts
                        path.requested_end = target_position
                        found_path = pathfinder.find_path(path)
                        if not found_path:
                            logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: cannot find path to target after initialization!")
                            break
                        path_points = path.points.copy()
                        travel_dist = path.geodesic_distance

                        # convert path points to normal and drop y-axis for tsdf planner
                        path_points = [pos_habitat_to_normal(p) for p in path_points]
                        path_points = [p[:2] for p in path_points]

                        # reset the max_explore_dist
                        max_explore_dist = travel_dist * cfg.max_step_dist_ratio
                        max_step = int(travel_dist * cfg.max_step_ratio) + cnt_step

                        state = 'explore'
                        explore_dist = 0.0

                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} initialization finished")

                    step_dict = {}
                    step_dict["agent_state"] = {}
                    step_dict["agent_state"]["init_pts"] = pts.tolist()
                    step_dict["agent_state"]["init_angle"] = rotation

                    logging.info(f"\n== step: {cnt_step}, {state} ==")

                    # for each position, get the views from different angles
                    if target_obj_id in tsdf_planner.simple_scene_graph.keys():
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
                    all_added_obj_ids = []
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

                        # construct an frequency count map of each semantic id to a unique id
                        obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                        added_obj_ids, annotated_image = tsdf_planner.update_scene_graph(
                            detection_model=detection_model,
                            rgb=rgb[..., :3],
                            semantic_obs=semantic_obs,
                            gt_obj_id_to_name=object_id_to_name,
                            gt_obj_id_to_bbox=object_id_to_bbox,
                            scannet_class_id_to_name=class_id_to_name,
                            cfg=cfg.scene_graph,
                            file_name=obs_file_name,
                            obs_point=pts,
                            return_annotated=True
                        )
                        # save the image as 720 x 720
                        plt.imsave(
                            os.path.join(object_feature_save_dir, obs_file_name),
                            np.asarray(Image.fromarray(rgb[..., :3]).resize((360, 360)))
                        )
                        all_added_obj_ids += added_obj_ids

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
                    all_added_obj_ids = [obj_id for obj_id in all_added_obj_ids if obj_id in tsdf_planner.simple_scene_graph]
                    # as well as the objects nearby
                    for obj_id, obj in tsdf_planner.simple_scene_graph.items():
                        if np.linalg.norm(obj.bbox_center[[0, 2]] - pts[[0, 2]]) < cfg.scene_graph.obj_include_dist + 0.5:
                            all_added_obj_ids.append(obj_id)
                    tsdf_planner.update_snapshots(obj_ids=set(all_added_obj_ids))
                    logging.info(f"Step {cnt_step} {len(tsdf_planner.simple_scene_graph)} objects, {len(tsdf_planner.snapshots)} snapshots")

                    update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
                    if not update_success:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: update frontier map failed!")
                        break

                    if target_found:
                        break

                    if len(tsdf_planner.frontiers) == 0 and state == 'initialize':
                        logging.info(f"Frontier exausted in initialization step {cnt_step}, directly go to explore state")
                        cnt_step = initialize_step - 1
                        continue

                    # Get observations for each frontier and store them
                    step_dict["frontiers"] = []
                    for i, frontier in enumerate(tsdf_planner.frontiers):
                        frontier_dict = {}
                        pos_voxel = frontier.position
                        pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                        pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                        frontier_dict["coordinate"] = pos_world.tolist()
                        # Turn to face the frontier point
                        if frontier.image is not None:
                            frontier_dict["rgb_id"] = frontier.image
                        else:
                            view_frontier_direction = np.asarray([pos_world[0] - pts[0], 0., pos_world[2] - pts[2]])

                            frontier_obs, target_detected = get_frontier_observation_and_detect_target(
                                agent, simulator, cfg.scene_graph, detection_model_yoloworld,
                                target_obj_id=target_obj_id, target_obj_class=object_id_to_name[target_obj_id],
                                view_frontier_direction=view_frontier_direction, init_pts=pts
                            )
                            if target_detected:
                                frontier.target_detected = True

                            # save the image as 720 x 720
                            plt.imsave(
                                os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                                np.asarray(Image.fromarray(frontier_obs).resize((360, 360)))
                            )
                            frontier.image = f"{cnt_step}_{i}.png"
                            frontier_dict["rgb_id"] = f"{cnt_step}_{i}.png"
                        step_dict["frontiers"].append(frontier_dict)

                    if state == 'explore':
                        # always reset target point to allow the model to choose again
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None
                    else:
                        # in initialization, always go to the target point
                        pass

                    # get the next choice
                    max_point_choice = None
                    if state == 'explore':
                        # choose the next frontier/snapshot accordingly
                        max_point_choice = tsdf_planner.get_next_choice(
                            pts=pts_normal,
                            path_points=path_points,
                            pathfinder=pathfinder,
                            target_obj_id=target_obj_id,
                            cfg=cfg.planner,
                        )
                        if max_point_choice is None:
                            logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: no valid choice!")
                            break
                    else:
                        # randomly choose a frontier as the max point in initialization
                        if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                            max_point_choice = random.choice(tsdf_planner.frontiers)
                            tsdf_planner.frontiers_weight = np.zeros(len(tsdf_planner.frontiers))
                            logging.info(f"Randomly choose frontier {max_point_choice.image}")

                    if max_point_choice is not None:
                        update_success = tsdf_planner.set_next_navigation_point(
                            choice=max_point_choice,
                            pts=pts_normal,
                            cfg=cfg.planner,
                            pathfinder=pathfinder,
                        )
                        if not update_success:
                            logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: set_next_navigation_point failed!")
                            break

                    return_values = tsdf_planner.agent_step(
                        pts=pts_normal,
                        angle=angle,
                        pathfinder=pathfinder,
                        cfg=cfg.planner,
                        path_points=path_points,
                        save_visualization=cfg.save_visualization,
                    )
                    if return_values[0] is None:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: find next navigation point failed!")
                        break
                    pts_normal, angle, pts_pix, fig, path_points, target_arrived = return_values

                    # save snapshots
                    step_dict["snapshots"] = format_snapshot(tsdf_planner.snapshots.values(), tsdf_planner.simple_scene_graph)

                    # sanity check
                    assert len(step_dict["snapshots"]) == len(tsdf_planner.snapshots), f"{len(step_dict['snapshots'])} != {len(tsdf_planner.snapshots)}"
                    obj_exclude_count = sum([1 if sum(obj.classes.values()) < 2 else 0 for obj in tsdf_planner.simple_scene_graph.values()])
                    total_objs_count = sum(
                        [len(snapshot.cluster) for snapshot in tsdf_planner.snapshots.values()]
                    )
                    assert len(tsdf_planner.simple_scene_graph) == total_objs_count + obj_exclude_count, f"{len(tsdf_planner.simple_scene_graph)} != {total_objs_count} + {obj_exclude_count}"
                    total_objs_count = sum(
                        [len(set(snapshot.cluster)) for snapshot in tsdf_planner.snapshots.values()]
                    )
                    assert len(tsdf_planner.simple_scene_graph) == total_objs_count + obj_exclude_count, f"{len(tsdf_planner.simple_scene_graph)} != {total_objs_count} + {obj_exclude_count}"
                    for obj_id in tsdf_planner.simple_scene_graph.keys():
                        exist_count = 0
                        for ss in tsdf_planner.snapshots.values():
                            if obj_id in ss.cluster:
                                exist_count += 1
                        if sum(tsdf_planner.simple_scene_graph[obj_id].classes.values()) < 2:
                            assert exist_count == 0, f"{exist_count} != 0 for obj_id {obj_id}"
                        else:
                            assert exist_count == 1, f"{exist_count} != 1 for obj_id {obj_id}"
                    for ss in tsdf_planner.snapshots.values():
                        assert len(ss.cluster) == len(set(ss.cluster)), f"{ss.cluster} has duplicates"
                        assert len(ss.full_obj_list.keys()) == len(set(ss.full_obj_list.keys())), f"{ss.full_obj_list.keys()} has duplicates"
                        for obj_id in ss.cluster:
                            assert obj_id in ss.full_obj_list, f"{obj_id} not in {ss.full_obj_list.keys()}"
                        for obj_id in ss.full_obj_list.keys():
                            assert obj_id in tsdf_planner.simple_scene_graph, f"{obj_id} not in scene graph"
                    # check whether the snapshots in scene.snapshots and scene.frames are the same
                    for file_name, ss in tsdf_planner.snapshots.items():
                        assert ss.cluster == tsdf_planner.frames[file_name].cluster, f"{ss}\n!=\n{tsdf_planner.frames[file_name]}"
                        assert ss.full_obj_list == tsdf_planner.frames[file_name].full_obj_list, f"{ss}\n==\n{tsdf_planner.frames[file_name]}"

                    if state == 'explore':
                        # save the ground truth choice
                        if type(max_point_choice) == SnapShot:
                            filename = max_point_choice.image
                            prediction = [float(ss["img_id"] == filename) for ss in step_dict["snapshots"]]
                            prediction += [0.0 for _ in range(len(step_dict["frontiers"]))]
                        elif type(max_point_choice) == Frontier:
                            prediction = [0.0 for _ in range(len(step_dict["snapshots"]))]
                            prediction += [float(ft_dict["rgb_id"] == max_point_choice.image) for ft_dict in step_dict["frontiers"]]
                        else:
                            raise ValueError("Invalid max_point_choice type")
                        assert len(prediction) == len(step_dict["snapshots"]) + len(step_dict["frontiers"]), f"{len(prediction)} != {len(step_dict['snapshots'])} + {len(step_dict['frontiers'])}"
                        if sum(prediction) != 1.0:
                            logging.info(f"Error! Prediction sum is not 1.0: {sum(prediction)}")
                            logging.info(max_point_choice)
                            logging.info(tsdf_planner.frontiers)
                            break
                        assert sum(prediction) == 1.0, f"{sum(prediction)} != 1.0"
                        step_dict["prediction"] = prediction

                        # record previous choice
                        step_dict["previous_choice"] = previous_choice_path  # this could be None or an image path of the frontier in last step
                        if type(max_point_choice) == Frontier:
                            previous_choice_path = max_point_choice.image

                        # sanity check
                        num_frontier = len(step_dict["frontiers"])
                        num_snapshot = len(step_dict["snapshots"])
                        chosen_idx = np.argwhere(np.array(step_dict["prediction"]) > 0.5).squeeze()
                        if chosen_idx >= num_snapshot:
                            chosen_idx -= num_snapshot
                            assert step_dict["frontiers"][chosen_idx]["rgb_id"] == max_point_choice.image, f"{step_dict['frontiers'][chosen_idx]['rgb_id']} != {max_point_choice.image}"
                            assert type(max_point_choice) == Frontier, f"{type(max_point_choice)} != Frontier"
                        else:
                            assert step_dict["snapshots"][chosen_idx]["img_id"] == max_point_choice.image, f"{step_dict['snapshots'][chosen_idx]['img_id']} != {max_point_choice.image}"
                            assert type(max_point_choice) == SnapShot, f"{type(max_point_choice)} != SnapShot"

                        # Save step data
                        if type(max_point_choice) == Frontier and tsdf_planner.frontiers_weight is not None and len(tsdf_planner.frontiers_weight) > 0 and np.max(tsdf_planner.frontiers_weight) < 1:
                            pass
                        else:
                            with open(os.path.join(episode_data_dir, f"{cnt_step:04d}.json"), "w") as f:
                                json.dump(step_dict, f, indent=4)

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
                                    img_path = os.path.join(object_feature_save_dir, max_point_choice.image)
                                    img = matplotlib.image.imread(img_path)
                                    axs[h_idx, w_idx].imshow(img)
                                    axs[h_idx, w_idx].set_title('Snapshot Chosen')
                        global_caption = f"{question_data['question']}\n{question_data['answer']}"
                        fig.suptitle(global_caption, fontsize=16)
                        plt.tight_layout(rect=(0., 0., 1., 0.95))
                        plt.savefig(os.path.join(frontier_video_path, f'{cnt_step}.png'))
                        plt.close()

                    if cfg.save_tsdf_video:
                        tsdf_video_path = os.path.join(episode_data_dir, "tsdf_video")
                        os.makedirs(tsdf_video_path, exist_ok=True)
                        num_images = len(tsdf_planner.snapshots.keys())
                        side_length = int(np.sqrt(num_images)) + 1
                        side_length = max(2, side_length)
                        fig, axs = plt.subplots(side_length, side_length, figsize=(40, 40))
                        snapshots = list(tsdf_planner.snapshots.keys())
                        for h_idx in range(side_length):
                            for w_idx in range(side_length):
                                axs[h_idx, w_idx].axis('off')
                                i = h_idx * side_length + w_idx
                                if i < num_images:
                                    img_path = os.path.join(object_feature_save_dir, tsdf_planner.snapshots[snapshots[i]].image)
                                    img = matplotlib.image.imread(img_path)
                                    axs[h_idx, w_idx].imshow(img)
                                    axs[h_idx, w_idx].set_title(f"Snapshot {i}")
                                    if type(max_point_choice) == SnapShot and max_point_choice.image == tsdf_planner.snapshots[snapshots[i]].image:
                                        axs[h_idx, w_idx].set_title(f"Snapshot {i} Chosen")
                        global_caption = f"{question_data['question']}\n{question_data['answer']}"
                        fig.suptitle(global_caption, fontsize=16)
                        plt.tight_layout(rect=(0., 0., 1., 0.95))
                        plt.savefig(os.path.join(tsdf_video_path, f'{cnt_step}.png'))
                        plt.close()

                    # update position and rotation
                    pts_normal = np.append(pts_normal, floor_height)
                    pts = pos_normal_to_habitat(pts_normal)
                    rotation = get_quaternion(angle, 0)
                    if type(max_point_choice) == Frontier:
                        # count the explore distance only when the agent is exploring, not approaching the target
                        explore_dist += np.linalg.norm(pts_pixs[-1] - pts_pixs[-2]) * tsdf_planner._voxel_size

                    logging.info(f"Current position: {pts}, {explore_dist:.3f}/{max_explore_dist:.3f}")

                    if type(max_point_choice) == SnapShot and target_arrived:
                        target_found = True
                        logging.info(f"Target observation position arrived at step {cnt_step}!")
                        # get an observation and break
                        agent_state.position = pts
                        agent_state.rotation = rotation
                        agent.set_state(agent_state)
                        obs = simulator.get_sensor_observations()
                        rgb = obs["color_sensor"]
                        target_obs_save_dir = os.path.join(episode_data_dir, "target_observation")
                        if cfg.save_visualization:
                            os.makedirs(target_obs_save_dir, exist_ok=True)
                            plt.imsave(os.path.join(target_obs_save_dir, f"{cnt_step}_target_observation.png"), rgb)
                        break

                if target_found:
                    metadata["episode_length"] = cnt_step
                    with open(os.path.join(episode_data_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=4)
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} finish with {cnt_step} steps")
                    success_count += 1
                    success_list.append(1)
                else:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} failed.")
                    if cfg.del_fail_case:
                        os.system(f"rm -r {episode_data_dir}")
                    success_list.append(0)
                
                logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")




                if os.path.exists(episode_data_dir):
                    # remove unused snapshots and frontiers
                    all_info_paths = glob.glob(os.path.join(episode_data_dir, "*.json"))
                    all_info_paths = [pth for pth in all_info_paths if 'metadata' not in pth]
                    all_snapshots = []
                    all_frontiers = []
                    for pth in all_info_paths:
                        step_data = json.load(open(pth, 'r'))
                        all_snapshots += [item['img_id'] for item in step_data['snapshots']]
                        all_frontiers += [item['rgb_id'] for item in step_data['frontiers']]
                    all_snapshots = set(all_snapshots)
                    all_frontiers = set(all_frontiers)
                    # remove unused snapshots
                    all_saved_snapshots = os.listdir(object_feature_save_dir)
                    for ss_id in all_saved_snapshots:
                        ss_step = int(ss_id.split('-')[0])
                        # only remove the unused snapshot collected during initialization
                        if ss_step < initialize_step and ss_id not in all_snapshots:
                            os.system(f"rm {os.path.join(object_feature_save_dir, ss_id)}")
                    # remove unused frontiers
                    all_saved_frontiers = os.listdir(episode_frontier_dir)
                    for ft_id in all_saved_frontiers:
                        if ft_id not in all_frontiers:
                            os.system(f"rm {os.path.join(episode_frontier_dir, ft_id)}")

                # print the stats of total number of images
                img_dict = {}
                for obj_id, obj in tsdf_planner.simple_scene_graph.items():
                    if obj.image not in img_dict:
                        img_dict[obj.image] = []
                    img_dict[obj.image].append(obj.object_id)
                total_images = len(img_dict.keys())
                total_images_record.append(total_images)
                total_success_images_record = [total_images_record[i] for i, s in enumerate(success_list) if s == 1]
                logging.info('\n')
                logging.info(total_images_record)
                logging.info(f"{question_data['question_id']}-path {path_idx} total images: {total_images}")
                logging.info(f"Average total images: {np.mean(total_images_record):.2f} +- {np.std(total_images_record):.2f}")
                logging.info(f"Max total images: {np.max(total_images_record)}, Min total images: {np.min(total_images_record)}")
                if len(total_success_images_record) > 0:
                    logging.info(f"Average success total images: {np.mean(total_success_images_record):.2f} +- {np.std(total_success_images_record):.2f}")
                    logging.info(f"Max total success images: {np.max(total_success_images_record)}, Min total success images: {np.min(total_success_images_record)}")


                filter_rank_all = json.load(open('data/selected_candidates.json', 'r'))
                filter_rank = filter_rank_all[question_data['question'] + '_' + scene_id]
                for top_k in [5, 10, 15, 20]:
                    top_k_classes = filter_rank[:top_k]
                    if object_id_to_name[target_obj_id] in top_k_classes:
                        top_k_correct_count[top_k] += 1

                    total_images = 0
                    for obj_list in img_dict.values():
                        for obj_id in obj_list:
                            if object_id_to_name[obj_id] in top_k_classes:
                                total_images += 1
                                break
                    top_k_images_record[top_k].append(total_images)
                    top_k_success_images_record = [top_k_images_record[top_k][i] for i, s in enumerate(success_list) if s == 1]
                    logging.info('\n')
                    logging.info(top_k_images_record[top_k])
                    logging.info(f"{question_data['question_id']}-path {path_idx} top {top_k} images: {total_images}")
                    logging.info(f"Average top {top_k} images: {np.mean(top_k_images_record[top_k]):.2f} +- {np.std(top_k_images_record[top_k]):.2f}")
                    logging.info(f"Max top {top_k} images: {np.max(top_k_images_record[top_k])}, Min top {top_k} images: {np.min(top_k_images_record[top_k])}")
                    logging.info(f"Top {top_k} correct count: {top_k_correct_count[top_k]}/{question_ind}")
                    if len(top_k_success_images_record) > 0:
                        logging.info(f"Average success top {top_k} images: {np.mean(top_k_success_images_record):.2f} +- {np.std(top_k_success_images_record):.2f}")
                        logging.info(f"Max top {top_k} success images: {np.max(top_k_success_images_record)}, Min top {top_k} success images: {np.min(top_k_success_images_record)}")

                # print the statistics of detection quality
                object_detection_rate = compute_recall(tsdf_planner.simple_scene_graph,object_id_to_name)
                logging.info(f"Object detection rate: {object_detection_rate:.2f}")

            logging.info(f"Question id {question_data['question_id']} finished all paths")

        logging.info(f'Scene {scene_id} finish')

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
    parser.add_argument("--path_id_offset", default=0, type=int)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--start", default=0.0, type=float)
    parser.add_argument("--end", default=1.0, type=float)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.cfg_file)
    OmegaConf.resolve(cfg)

    # Set up logging
    cfg.output_dir = os.path.join(cfg.output_parent_dir, cfg.exp_name)
    if not os.path.exists(cfg.output_dir):
        os.makedirs(cfg.output_dir, exist_ok=True)  # recursive
    if not os.path.exists(cfg.dataset_output_dir):
        os.makedirs(cfg.dataset_output_dir, exist_ok=True)
    logging_path = os.path.join(cfg.dataset_output_dir, "log.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(logging_path, mode="w"),
            logging.StreamHandler(),
        ],
    )

    cfg.path_id_offset = args.path_id_offset
    if args.seed is not None:
        cfg.seed = args.seed

    if args.start >= args.end:
        raise ValueError(f"Start {args.start} should be less than end {args.end}")

    # run
    logging.info(f"***** Running {cfg.exp_name} *****")
    main(cfg)
