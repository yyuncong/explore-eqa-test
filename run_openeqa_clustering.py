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
from src.tsdf_clustering import TSDFPlanner, Frontier, SnapShot
from inference.models import YOLOWorld
import torch


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

    scene_pose_path = cfg.scene_pose_path
    scene_pose_map = {
        pose_folder.split("-")[-1]: os.listdir(
            os.path.join(scene_pose_path, pose_folder)
        ) for pose_folder in os.listdir(scene_pose_path)
    }
    scene_id_map = {
        pose_folder.split("-")[-1]: pose_folder
        for pose_folder in os.listdir(scene_pose_path)
    }

    # for each scene, answer each question
    question_ind = 0
    success_count = 0

    # hack for saving runing time
    scene_obs_map = {}

    max_num_snapshots = 0
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data['question_id']
        question = question_data['question']
        answer = question_data['answer']

        # Extract question
        scene_id = question_data["episode_history"]

        if scene_id in scene_obs_map:
            logging.info(f"Scene {scene_id} already processed")
            os.system(f"cp -r {os.path.join(cfg.output_dir, str(scene_obs_map[scene_id]))} {os.path.join(cfg.output_dir, str(question_id))}")
            continue

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
        # episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_observations_dir, exist_ok=True)
        os.makedirs(episode_object_observe_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)
        # os.makedirs(episode_snapshot_dir, exist_ok=True)

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
        scene_pose_list = scene_pose_map[scene_id.split("-")[-1]]
        scene_pose_list = sorted(scene_pose_list, key=lambda x: int(x.split(".")[0]))
        for i in range(len(scene_pose_list)):
            pose_file = scene_pose_list[i]
            with open(os.path.join(scene_pose_path, scene_id_map[scene_id.split("-")[-1]], pose_file), "rb") as f:
                pose_data = pickle.load(f)
            agent_state = pose_data["agent_state"]
            agent.set_state(agent_state)
            pts = agent_state.position
            rotation = agent_state.rotation
            angle, axis = quat_to_angle_axis(rotation)
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
            obs_file_name = f"{i}.png"
            with torch.no_grad():
                object_added, annotated_image = tsdf_planner.update_scene_graph(
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

            if cfg.save_egocentric_view:
                plt.imsave(os.path.join(episode_observations_dir, obs_file_name), rgb)

            if not cfg.incremental:
                tsdf_planner.update_snapshots(
                    obj_id_to_bbox=object_id_to_bbox,
                    obj_set=set(tsdf_planner.scene_graph_list),
                    incremental=cfg.incremental,
                )
            else:
                # find the object ids that are within 2.5m away from the agent
                obj_ids = tsdf_planner.scene_graph_list.copy()[
                    tsdf_planner.prev_scene_graph_length:
                ]
                for obj_id, obj in tsdf_planner.simple_scene_graph.items():
                    if np.linalg.norm(obj.bbox_center - pts) < cfg.scene_graph.obj_include_dist:
                        obj_ids.append(obj_id)
                tsdf_planner.update_snapshots(
                    obj_id_to_bbox=object_id_to_bbox,
                    obj_set=set(obj_ids),
                    incremental=cfg.incremental,
                )
            tsdf_planner.prev_scene_graph_length = len(tsdf_planner.scene_graph_list)
            logging.info(f"Step {i} total objects: {len(tsdf_planner.simple_scene_graph)}, total snapshots: {len(tsdf_planner.snapshots)}")

            total_objs_count = sum(
                [len(snapshot.cluster) for snapshot in tsdf_planner.snapshots.values()]
            )
            assert len(tsdf_planner.simple_scene_graph) == total_objs_count, f"{len(tsdf_planner.simple_scene_graph)} != {total_objs_count}"
            total_objs_count = sum(
                [len(set(snapshot.cluster)) for snapshot in tsdf_planner.snapshots.values()]
            )
            assert len(tsdf_planner.simple_scene_graph) == total_objs_count, f"{len(tsdf_planner.simple_scene_graph)} != {total_objs_count}"
            for obj_id in tsdf_planner.simple_scene_graph.keys():
                exist_count = 0
                for ss in tsdf_planner.snapshots.values():
                    if obj_id in ss.cluster:
                        exist_count += 1
                assert exist_count == 1, f"{exist_count} != 1 for obj_id {obj_id}, {object_id_to_name[obj_id]}"
            for ss in tsdf_planner.snapshots.values():
                assert len(ss.cluster) == len(set(ss.cluster)), f"{ss.cluster} has duplicate objects"
                assert len(ss.full_obj_list.keys()) == len(set(ss.full_obj_list.keys())), f"{ss.full_obj_list} has duplicate objects"
                for obj_id in ss.cluster:
                    assert obj_id in ss.full_obj_list, f"{obj_id} not in {ss.full_obj_list}"
                for obj_id in ss.full_obj_list.keys():
                    assert obj_id in tsdf_planner.simple_scene_graph, f"{obj_id} not in scene graph"

        logging.info(f"{question_idx}/{total_questions}")

        # TODO: save snapshots here
        if cfg.save_snapshots:
            snapshots_path = os.path.join(episode_data_dir, "snapshots")
            os.makedirs(snapshots_path, exist_ok=True)
            num_images = len(tsdf_planner.snapshots.keys())
            snapshots = list(tsdf_planner.snapshots.keys())
            for snapshot in snapshots:
                os.system(
                    f"cp {episode_observations_dir}/{snapshot} {snapshots_path}/{snapshot}"
                )
        # hack to save processing time
        scene_obs_map[scene_id] = question_id
        max_num_snapshots = max(max_num_snapshots, len(tsdf_planner.snapshots))

    logging.info(f'All scenes finish')
    logging.info(f'Maximum number of snapshots: {max_num_snapshots}')
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
    # if not os.path.exists(cfg.dataset_output_dir):
    #     os.makedirs(cfg.dataset_output_dir, exist_ok=True)
    logging_path = os.path.join(cfg.output_dir, "log.log")
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
