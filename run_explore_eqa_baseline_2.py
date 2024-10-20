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
import math
import quaternion
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import habitat_sim
from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors, quat_to_angle_axis
from src.habitat import (
    make_simple_cfg,
    make_semantic_cfg,
    pos_normal_to_habitat,
    pos_habitat_to_normal,
    pose_habitat_to_normal,
    pose_normal_to_tsdf,
    get_quaternion,
    get_navigable_point_to
)
from src.geom import get_cam_intr, get_scene_bnds, get_collision_distance
from src.vlm import VLM
from src.tsdf_original_2 import TSDFPlanner, Frontier, Object
from inference.models import YOLOWorld

'''
Baseline 2:
Use explore-eqa like vlm prompting approach on my frontier exploration algorithm.
'''


def main(cfg):
    camera_tilt = cfg.camera_tilt_deg * np.pi / 180
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model = YOLOWorld(model_id=cfg.detection_model_name)

    # Load VLM
    vlm = VLM(cfg.vlm)

    # Load dataset
    all_paths_list = os.listdir(cfg.path_data_dir)
    all_paths_list = [question_id for question_id in all_paths_list if 650 < int(question_id.split('-')[0]) < 750]

    total_questions = 0
    success_count = 0
    total_explore_distance = 0

    # Run all questions
    for question_idx in tqdm(range(len(all_paths_list))):
        total_questions += 1
        question_id = all_paths_list[question_idx]
        metadata = json.load(open(os.path.join(cfg.path_data_dir, question_id, "metadata.json"), "r"))

        # Extract question
        scene_id = metadata["scene"]
        question = metadata["question"]
        init_pts = metadata["init_pts"]
        init_angle = metadata["init_angle"]
        target_obj_id = metadata["target_obj_id"]
        target_obj_class = metadata["target_obj_class"]
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id} Target: {target_obj_class}")

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
        }
        sim_cfg = make_semantic_cfg(sim_settings)
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

        obj_bbox = [item['bbox'] for item in bounding_box_data if int(item['id']) == target_obj_id][0]
        obj_bbox = np.asarray(obj_bbox)  # (2, 3)
        obj_bbox_center = np.mean(obj_bbox, axis=0)
        obj_bbox_center = obj_bbox_center[[0, 2, 1]]

        episode_data_dir = os.path.join(str(cfg.output_dir), str(question_id))
        episode_frontier_dir = os.path.join(str(cfg.frontier_dir), str(question_id))
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)

        pts = init_pts
        angle = init_angle

        # Floor - use pts height as floor height
        rotation = get_quaternion(angle, camera_tilt)

        pts_normal = pos_habitat_to_normal(pts)
        floor_height = pts_normal[-1]
        tsdf_bnds, scene_size = get_scene_bnds(pathfinder, floor_height)
        num_step = int(math.sqrt(scene_size) * cfg.max_step_room_size_ratio)
        logging.info(
            f"Scene size: {scene_size} Floor height: {floor_height} Steps: {num_step}"
        )
        try:
            del tsdf_planner
        except:
            pass
        # Initialize TSDF
        tsdf_planner = TSDFPlanner(
            vol_bnds=tsdf_bnds,
            voxel_size=cfg.tsdf_grid_size,
            floor_height_offset=0,
            pts_init=pos_habitat_to_normal(pts),
            init_clearance=cfg.init_clearance * 2,
        )

        target_center_voxel = tsdf_planner.world2vox(pos_habitat_to_normal(obj_bbox_center))
        pts_pixs = np.empty((0, 2))  # for plotting path on the image
        pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

        # Run steps
        target_found = False
        cnt_step = -1
        explore_distance = 0
        prev_pts = pts.copy()
        while cnt_step < num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")

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

            unoccupied_map, _ = tsdf_planner.get_island_around_pts(
                pts_normal, height=1.2
            )
            occupied_map = np.logical_not(unoccupied_map)

            # observe and update the TSDF
            for view_idx, ang in enumerate(all_angles):
            # logging.info(f"Step {cnt_step}, view {view_idx + 1}/{total_views}")

                # check whether current view is valid
                collision_dist = tsdf_planner._voxel_size * get_collision_distance(
                    occupied_map,
                    pos=tsdf_planner.habitat2voxel(pts),
                    direction=tsdf_planner.rad2vector(ang)
                )
                if collision_dist < cfg.collision_dist and view_idx != total_views - 1:  # the last view is the main view, and is not dropped
                    # logging.info(f"Collision detected at step {cnt_step} view {view_idx}")
                    continue

                agent_state.position = pts
                agent_state.rotation = get_quaternion(ang, camera_tilt)
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

                # check whether the observation is valid
                keep_observation = True
                black_pix_ratio = np.sum(semantic_obs == 0) / (img_height * img_width)
                if black_pix_ratio > cfg.black_pixel_ratio:
                    keep_observation = False
                positive_depth = depth[depth > 0]
                if positive_depth.size == 0 or np.percentile(positive_depth, 30) < cfg.min_30_percentile_depth:
                    keep_observation = False
                if not keep_observation and view_idx != total_views - 1:
                    # logging.info(f"Invalid observation: black pixel ratio {black_pix_ratio}, 30 percentile depth {np.percentile(depth[depth > 0], 30)}")
                    continue

                # construct an frequency count map of each semantic id to a unique id
                target_in_view, annotated_rgb = tsdf_planner.update_scene_graph(
                    detection_model=detection_model,
                    rgb=rgb[..., :3],
                    semantic_obs=semantic_obs,
                    obj_id_to_name=object_id_to_name,
                    obj_id_to_bbox=object_id_to_bbox,
                    cfg=cfg.scene_graph,
                    target_obj_id=target_obj_id,
                    return_annotated=True
                )

                # check stop condition
                if target_in_view:
                    if target_obj_id in tsdf_planner.simple_scene_graph.keys():
                        target_obj_pix_ratio = np.sum(semantic_obs == target_obj_id) / (img_height * img_width)
                        if target_obj_pix_ratio > 0:
                            obj_pix_center = np.mean(np.argwhere(semantic_obs == target_obj_id), axis=0)
                            bias_from_center = (obj_pix_center - np.asarray([img_height // 2, img_width // 2])) / np.asarray([img_height, img_width])
                            # currently just consider that the object should be in around the horizontal center, not the vertical center
                            # due to the viewing angle difference
                            if target_obj_pix_ratio > cfg.stop_min_pix_ratio and np.abs(bias_from_center)[1] < cfg.stop_max_bias_from_center:
                                logging.info(f"Stop condition met at step {cnt_step} view {view_idx}")
                                target_found = True

                # TSDF fusion
                tsdf_planner.integrate(
                    color_im=rgb,
                    depth_im=depth,
                    cam_intr=cam_intr,
                    cam_pose=cam_pose_tsdf,
                    obs_weight=1.0,
                    margin_h=int(cfg.margin_h_ratio * img_height),
                    margin_w=int(cfg.margin_w_ratio * img_width),
                )

                if cfg.save_obs:
                    if target_found:
                        plt.imsave(os.path.join(episode_data_dir, f"{cnt_step}-view_{view_idx}-target.png"), annotated_rgb)
                    else:
                        plt.imsave(os.path.join(episode_data_dir, f"{cnt_step}-view_{view_idx}.png"), annotated_rgb)

                if target_found:
                    break

            if target_found:
                break

            update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
            if not update_success:
                logging.info(f"Question id {scene_id} invalid: update frontier map failed!")
                break

            # Turn to face each frontier point and get rgb image, and a score from vlm
            for i, frontier in enumerate(tsdf_planner.frontiers):
                pos_voxel = frontier.position
                pos_world = pos_voxel * tsdf_planner._voxel_size + tsdf_planner._vol_origin[:2]
                pos_world = pos_normal_to_habitat(np.append(pos_world, floor_height))
                # Turn to face the frontier point
                if frontier.image is not None:
                    original_path = os.path.join(episode_frontier_dir, frontier.image)
                    if os.path.exists(original_path):
                        target_path = os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png")
                        os.system(f"cp {original_path} {target_path}")
                    assert frontier.score != 0, f'{frontier.image}, {frontier.score}'
                else:
                    view_frontier_direction = np.asarray([pos_world[0] - pts[0], 0., pos_world[2] - pts[2]])
                    default_view_direction = np.asarray([0., 0., -1.])
                    if np.linalg.norm(view_frontier_direction) < 1e-3:
                        view_frontier_direction = default_view_direction
                    if np.dot(view_frontier_direction, default_view_direction) / np.linalg.norm(view_frontier_direction) < -1 + 1e-3:
                        # if the rotation is to rotate 180 degree, then the quaternion is not unique
                        # we need to specify rotating along y-axis
                        agent_state.rotation = quat_to_coeffs(
                            quaternion.quaternion(0, 0, 1, 0)
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                    else:
                        agent_state.rotation = quat_to_coeffs(
                            quat_from_two_vectors(default_view_direction, view_frontier_direction)
                            * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
                        ).tolist()
                    agent.set_state(agent_state)
                    # Get observation at current pose - skip black image, meaning robot is outside the floor
                    obs = simulator.get_sensor_observations()
                    rgb = obs["color_sensor"]
                    # plt.imsave(
                    #     os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                    #     rgb,
                    # )
                    frontier.image = f"{cnt_step}_{i}.png"

                    # Get score from VLM using rgb
                    rgb_im = Image.fromarray(rgb, mode="RGBA").convert("RGB")
                    prompt = f"Consider the question: {question}, and you will explore the area in the image. Is the direction in the image worth exploring for answering the question? Answer with Yes or No."
                    frontier.score = vlm.get_loss(rgb_im, prompt, ["Yes", "No"])[0]

                    if cfg.save_frontier:
                        # write the question and score on the image
                        draw = ImageDraw.Draw(rgb_im)
                        font = ImageFont.load_default()
                        draw.text((10, 10), f"Question: {question}", (255, 255, 255), font=font)
                        draw.text((10, 30), f"Score: {frontier.score}", (255, 255, 255), font=font)
                        rgb_im.save(os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"))

            max_point_choice = tsdf_planner.get_next_choice(
                pts=pts_normal,
                angle=angle,
                cfg=cfg.planner,
            )
            if max_point_choice is None:
                logging.info(f"Question {scene_id} invalid: no valid choice!")
                break

            return_values = tsdf_planner.get_next_navigation_point(
                choice=max_point_choice,
                pts=pts_normal,
                angle=angle,
                pathfinder=pathfinder,
                cfg=cfg.planner,
                save_visualization=cfg.save_visualization,
            )
            if return_values[0] is None:
                logging.info(f"Question id {scene_id} invalid: find next navigation point failed!")
                break
            pts_normal, angle, pts_pix, fig = return_values

            pts_pixs = np.vstack((pts_pixs, pts_pix))
            if cfg.save_visualization:
                # Add path to ax5, with colormap to indicate order
                ax5 = fig.axes[4]
                ax5.plot(pts_pixs[:, 1], pts_pixs[:, 0], linewidth=5, color="black")
                ax5.scatter(pts_pixs[0, 1], pts_pixs[0, 0], c="white", s=50)

                # add target object bbox
                color = 'green' if target_obj_id in tsdf_planner.simple_scene_graph.keys() else 'red'
                ax5.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s = 120)
                ax1, ax2, ax4 = fig.axes[0], fig.axes[1], fig.axes[3]
                ax4.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s = 120)
                ax1.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s = 120)
                ax2.scatter(target_center_voxel[1], target_center_voxel[0], c=color, s = 120)

                fig.tight_layout()
                plt.savefig(os.path.join(episode_data_dir, "{}_map.png".format(cnt_step)))
                plt.close()

            # update position and rotation
            pts_normal = np.append(pts_normal, floor_height)
            pts = pos_normal_to_habitat(pts_normal)
            rotation = get_quaternion(angle, camera_tilt)

            # update distance
            explore_distance += np.linalg.norm(pts - prev_pts)
            prev_pts = pts.copy()

        if target_found:
            success_count += 1
            total_explore_distance += explore_distance

        logging.info(f"Success rate: {success_count}/{total_questions} = {success_count / total_questions}")
        logging.info(f"Average explore distance for successful episodes: {total_explore_distance / success_count}")



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
    cfg.frontier_dir = os.path.join(cfg.output_dir, "frontier")
    if not os.path.exists(cfg.frontier_dir):
        os.makedirs(cfg.frontier_dir, exist_ok=True)  # recursive
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
