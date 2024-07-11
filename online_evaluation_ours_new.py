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
from src.eval_utils import prepare_step_dict, get_item, encode, load_scene_features, rgba2rgb, load_checkpoint, collate_wrapper, construct_selection_prompt
from inference.models import YOLOWorld

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

from easydict import EasyDict

def main(cfg):
    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model = YOLOWorld(model_id=cfg.detection_model_name)

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

    print("load model")
    # Initialize LLaVA model
    # model_path = "liuhaotian/llava-v1.5-7b"
    model_path = "/work/pi_chuangg_umass_edu/yuncong/llava-v1.5-7b"
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, None, model_name, device_map=None, add_multisensory_token=True)  
    # model = model.to("cuda")
    load_checkpoint(model, cfg.model_path)
    model = model.to("cuda")
    # model = None
    model.eval()
    print("finish loading model")

    # for each scene, answer each question
    question_ind = 0
    success_count = 0
    same_class_count = 0

    success_list = []
    path_length_list = []
    dist_from_chosen_to_target_list = []

    for scene_id in all_scene_list:
        if '00723' in scene_id:
            continue

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

        # load scene
        scene_path = cfg.scene_data_path_train if int(scene_id.split("-")[0]) < 800 else cfg.scene_data_path_val
        scene_features_path = cfg.scene_features_path_train if int(scene_id.split("-")[0]) < 800 else cfg.scene_features_path_val
        scene_mesh_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(scene_path, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        bbox_data_path = os.path.join(cfg.semantic_bbox_data_path, scene_id + ".json")
        # assert os.path.exists(scene_mesh_path) and os.path.exists(navmesh_path), f'{scene_mesh_path}, {navmesh_path}'
        # assert os.path.exists(semantic_texture_path) and os.path.exists(scene_semantic_annotation_path), f'{semantic_texture_path}, {scene_semantic_annotation_path}'
        if not os.path.exists(scene_mesh_path) or not os.path.exists(navmesh_path) or not os.path.exists(semantic_texture_path) or not os.path.exists(scene_semantic_annotation_path):
            logging.info(f"Scene {scene_id} not found, skip")
            continue
        if not os.path.exists(bbox_data_path):
            logging.info(f"Scene {scene_id} bbox data not found, skip")
            continue
        if not os.path.exists(scene_features_path):
            logging.info(f"Scene {scene_id} features not found, skip")
            continue

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
        bounding_box_data = json.load(open(bbox_data_path, "r"))
        object_id_to_bbox = {int(item['id']): {'bbox': item['bbox'], 'class': item['class_name']} for item in bounding_box_data}
        object_id_to_name = {int(item['id']): item['class_name'] for item in bounding_box_data}

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
            episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
            object_feature_save_dir = os.path.join(episode_data_dir, 'object_features')
            os.makedirs(episode_data_dir, exist_ok=True)
            os.makedirs(episode_frontier_dir, exist_ok=True)
            os.makedirs(object_feature_save_dir, exist_ok=True)

            pts = init_pts
            angle = init_angle
            rotation = get_quaternion(angle, 0)

            # initialize the TSDF
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

            logging.info(f'\n\nQuestion id {scene_id} initialization successful!')

            # run steps
            target_found = False
            explore_dist = 0.0
            cnt_step = -1
            dist_from_chosen_to_target = None
            while cnt_step < num_step - 1:
                cnt_step += 1
                logging.info(f"\n== step: {cnt_step}")
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

                    rgb = rgba2rgb(rgb)
                    rgb_egocentric_views.append(rgb)

                    # construct an frequency count map of each semantic id to a unique id
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
                        plt.imsave(os.path.join(object_feature_save_dir, obs_file_name), rgb)

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

                        frontier_obs = get_frontier_observation(
                            agent, simulator, cfg, tsdf_planner,
                            view_frontier_direction=view_frontier_direction,
                            init_pts=pts,
                            camera_tilt=0,
                            max_try_count=0
                        )

                        plt.imsave(
                            os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                            frontier_obs,
                        )
                        processed_rgb = rgba2rgb(frontier_obs)
                        with torch.no_grad():
                            img_feature = encode(model, image_processor, processed_rgb).mean(1)
                        assert img_feature is not None
                        frontier.image = f"{cnt_step}_{i}.png"
                        frontier.feature = img_feature

                if cfg.choose_every_step:
                    if tsdf_planner.max_point is not None and type(tsdf_planner.max_point) == Frontier:
                        # reset target point to allow the model to choose again
                        tsdf_planner.max_point = None
                        tsdf_planner.target_point = None

                if tsdf_planner.max_point is None and tsdf_planner.target_point is None:
                    # TODO: get the max_point_choice from the model
                    max_point_choice: [Frontier, SnapShot] = None

                    if max_point_choice is None:
                        logging.info(f"Question id {question_id} invalid: no valid choice!")
                        break

                    update_success = tsdf_planner.set_next_navigation_point(
                        choice=max_point_choice,
                        pts=pts_normal,
                        cfg=cfg.planner,
                        pathfinder=pathfinder,
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

                    # add target object bbox
                    color = 'green' if target_obj_id in tsdf_planner.simple_scene_graph.keys() else 'red'
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
                                img_path = os.path.join(object_feature_save_dir, max_point_choice.image)
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
                    # the model found a target snapshot and arrived at a proper observation point
                    # get an observation and save it
                    agent_state_obs = habitat_sim.AgentState()
                    agent_state_obs.position = pts
                    agent_state_obs.rotation = rotation
                    agent.set_state(agent_state_obs)
                    obs = simulator.get_sensor_observations()
                    rgb = obs["color_sensor"]
                    target_obs_save_dir = os.path.join(episode_data_dir, "target_observation")
                    os.makedirs(target_obs_save_dir, exist_ok=True)
                    plt.imsave(os.path.join(target_obs_save_dir, f"target.png"), rgb)

                    # get some statistics

                    # check whether the target object is in the selected snapshot
                    if target_obj_id in tsdf_planner.snapshots[max_point_choice.image].selected_obj_list:
                        logging.info(f"{target_obj_id} in Chosen snapshot {max_point_choice.image}! Success!")
                        target_found = True
                    else:
                        logging.info(f"Question id {question_id} choose the wrong snapshot! Failed!")

                    # check whether the class of the target object is the same as the class of the selected snapshot
                    for ss_obj_id in tsdf_planner.snapshots[max_point_choice.image].selected_obj_list:
                        if object_id_to_name[ss_obj_id] == object_id_to_name[target_obj_id]:
                            same_class_count += 1
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
            path_length_list.append(explore_dist)
            if dist_from_chosen_to_target is not None:
                dist_from_chosen_to_target_list.append(dist_from_chosen_to_target)

            logging.info(f"\n#######################################################")
            logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")
            logging.info(f"Same class ratio: {same_class_count}/{question_ind}")
            logging.info(f"Mean path length for success exploration: {np.mean([x for i, x in enumerate(path_length_list) if success_list[i] == 1])}")
            logging.info(f"Mean path length for all exploration: {np.mean(path_length_list)}")
            logging.info(f"Mean distance from final position to target observation position: {np.mean(dist_from_chosen_to_target_list)}")
            logging.info(f"#######################################################\n")

        logging.info(f'Scene {scene_id} finish')

    result_dict = {}
    result_dict["success_rate"] = success_count / question_ind
    result_dict["same_class_ratio"] = same_class_count / question_ind
    result_dict["mean_path_length"] = np.mean(path_length_list)
    result_dict["mean_success_path_length"] = np.mean([x for i, x in enumerate(path_length_list) if success_list[i] == 1])
    result_dict["mean_distance_from_chosen_to_target"] = np.mean(dist_from_chosen_to_target_list)
    with open(os.path.join(cfg.output_dir, "result.json"), "w") as f:
        json.dump(result_dict, f, indent=4)

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
