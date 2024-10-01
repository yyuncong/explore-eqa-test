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
from src.eval_utils_snapshot_new import rgba2rgb
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
    questions_list = json.load(open(cfg.questions_list_path, "r"))
    total_questions = len(questions_list)
    # sort the data according to the question id
    questions_list = sorted(questions_list, key=lambda x: x['question_id'])
    logging.info(f"Total number of questions: {total_questions}")
    questions_list = questions_list[int(start_ratio * total_questions):int(end_ratio * total_questions)]
    # shuffle the data
    # random.shuffle(questions_list)
    logging.info(f"number of questions after splitting: {len(questions_list)}")
    logging.info(f"question path: {cfg.questions_list_path}")

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

    # load success list and path length list
    if os.path.exists(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl")):
        with open(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl"), "rb") as f:
            success_list = pickle.load(f)
    else:
        success_list = []
    if os.path.exists(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl")):
        with open(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl"), "rb") as f:
            path_length_list = pickle.load(f)
    else:
        path_length_list = {}
    if os.path.exists(os.path.join(str(cfg.output_dir), f"fail_list_{start_ratio}_{end_ratio}.pkl")):
        with open(os.path.join(str(cfg.output_dir), f"fail_list_{start_ratio}_{end_ratio}.pkl"), "rb") as f:
            fail_list = pickle.load(f)
    else:
        fail_list = []
    if os.path.exists(os.path.join(str(cfg.output_dir), f"gpt_answer_{start_ratio}_{end_ratio}.json")):
        with open(os.path.join(str(cfg.output_dir), f"gpt_answer_{start_ratio}_{end_ratio}.json"), "r") as f:
            gpt_answer_list = json.load(f)
    else:
        gpt_answer_list = []
    if os.path.exists(os.path.join(str(cfg.output_dir), f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json")):
        with open(os.path.join(str(cfg.output_dir), f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"), "r") as f:
            n_filtered_snapshots_list = json.load(f)
    else:
        n_filtered_snapshots_list = {}
    if os.path.exists(os.path.join(str(cfg.output_dir), f"n_total_snapshots_{start_ratio}_{end_ratio}.json")):
        with open(os.path.join(str(cfg.output_dir), f"n_total_snapshots_{start_ratio}_{end_ratio}.json"), "r") as f:
            n_total_snapshots_list = json.load(f)
    else:
        n_total_snapshots_list = {}
    if os.path.exists(os.path.join(str(cfg.output_dir), f"n_total_frames_{start_ratio}_{end_ratio}.json")):
        with open(os.path.join(str(cfg.output_dir), f"n_total_frames_{start_ratio}_{end_ratio}.json"), "r") as f:
            n_total_frames_list = json.load(f)
    else:
        n_total_frames_list = {}

    success_count = 0
    max_target_observation = cfg.max_target_observation

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data['question_id']
        question = question_data['question']
        answer = question_data['answer']

        if question_id in success_list or question_id in fail_list:
            logging.info(f"Question {question_id} already processed")
            success_count += 1
            continue

        # Extract question
        scene_id = question_data["episode_history"]

        if '00853' in scene_id:
            logging.info(f"Skip scene 00853")
            continue

        init_pts = question_data["position"]
        init_quat = quaternion.quaternion(*question_data["rotation"])
        logging.info(f"\n========\nIndex: {question_idx} Scene: {scene_id}")

        # load scene
        try:
            del scene
        except:
            pass

        scene = Scene(scene_id, cfg, cfg_cg)

        # Set the classes for the detection model
        detection_model.set_classes(scene.obj_classes.get_classes_arr())

        episode_data_dir = os.path.join(str(cfg.output_dir), str(question_id))
        episode_object_observe_dir = os.path.join(episode_data_dir, 'object_observations')
        episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
        episode_snapshot_dir = os.path.join(episode_data_dir, 'snapshot')
        os.makedirs(episode_data_dir, exist_ok=True)
        os.makedirs(episode_object_observe_dir, exist_ok=True)
        os.makedirs(episode_frontier_dir, exist_ok=True)
        os.makedirs(episode_snapshot_dir, exist_ok=True)

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

        # record the history of the agent's path
        pts_pixs = np.empty((0, 2))
        pts_pixs = np.vstack((pts_pixs, tsdf_planner.habitat2voxel(pts)[:2]))

        logging.info(f'\n\nQuestion id {question_id} initialization successful!')

        # run steps
        target_found = False
        explore_dist = 0.0
        cnt_step = -1

        all_snapshots = {}
        all_target_observations = []
        gpt_answer = None
        n_filtered_snapshots = 0
        n_total_snapshots = 0
        n_total_frames = 0
        while cnt_step < num_step - 1:
            cnt_step += 1
            logging.info(f"\n== step: {cnt_step}")
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
                obs, cam_pose = scene.get_observation(pts, ang)
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]
                rgb = rgba2rgb(rgb)

                cam_pose_normal = pose_habitat_to_normal(cam_pose)
                cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                # collect all view features
                obs_file_name = f"{cnt_step}-view_{view_idx}.png"
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
                    resized_rgb = resize_image(rgb, cfg.prompt_h, cfg.prompt_w)
                    all_snapshots[obs_file_name] = resized_rgb
                    rgb_egocentric_views.append(resized_rgb)
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
            scene.update_snapshots(obj_ids=set(all_added_obj_ids), min_detection=cfg.min_detection)
            logging.info(f"Step {cnt_step} {len(scene.objects)} objects, {len(scene.snapshots)} snapshots")
            n_total_snapshots = len(scene.snapshots)
            n_total_frames = len(scene.frames)

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
                if cnt_step == 0:  # if the first step fails, we should stop
                    logging.info(f"Question id {question_id} invalid: update_frontier_map failed!")
                    break

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
                            os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                            frontier_obs,
                        )
                    processed_rgb = resize_image(rgba2rgb(frontier_obs), cfg.prompt_h, cfg.prompt_w)
                    frontier.image = f"{cnt_step}_{i}.png"
                    frontier.feature = processed_rgb  # yh: in gpt4-based exploration, feature is no used. So I just directly use this attribute to store raw rgb. Otherwise, the additional .img attribute may cause bugs.

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
                step_dict["question"] = question
                step_dict["scene"] = scene_id

                outputs, snapshot_id_mapping, reason, n_filtered_snapshots = explore_step(step_dict, cfg)
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

                    # add the reason for the choice
                    gpt_answer = reason  # use the latest reason

                    # for snapshot_id, obj_list in snapshot_dict.items():
                    #     logging.info(f"{snapshot_id}:")
                    #     for obj_str in obj_list:
                    #         logging.info(f"\t{obj_str}")
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
                    logging.info(f"Question id {question_id} invalid: no valid choice!")
                    break

                update_success = tsdf_planner.set_next_navigation_point(
                    choice=max_point_choice,
                    pts=pts_normal,
                    objects=scene.objects,
                    cfg=cfg.planner,
                    pathfinder=scene.pathfinder,
                    random_position=False,
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

            logging.info(f"Current position: {pts}, {explore_dist:.3f}")

            if type(max_point_choice) == SnapShot:
                # get an observation and break
                obs, _ = scene.get_observation(pts, angle)
                rgb = obs["color_sensor"]
                rgb = rgba2rgb(rgb)
                all_target_observations.append(rgb)

                # save the snapshot image itself
                snapshot_filename = max_point_choice.image.split(".")[0]
                os.system(f"cp {os.path.join(episode_snapshot_dir, max_point_choice.image)} {os.path.join(episode_object_observe_dir, f'snapshot_{snapshot_filename}.png')}")

                if target_arrived:
                    target_found = True
                    logging.info(f"Question id {question_id} finished after arriving at target! In total {len(all_target_observations)} target observations")
                    break

        
            # if agent postion is within 1m of the snapshot position then target_found is True
            # calculate the target position by averaging all objects' positions in max_point_choice.cluster                

        if len(all_target_observations) > 0:
            target_found = True

        # save the last "max_target_observation" target observations
        for i, target_observation in enumerate(all_target_observations[-max_target_observation:]):
            plt.imsave(os.path.join(episode_object_observe_dir, f"target_observation_{i}.png"), target_observation)

        if target_found:
            success_count += 1
            # We only consider samples that model predicts object (use baseline results other samples for now)
            # TODO: you can save path_length in the same format as you did for the baseline
            if question_id not in success_list:
                success_list.append(question_id)
            path_length_list[question_id] = explore_dist
            logging.info(f"Question id {question_id} finish with {cnt_step} steps, {explore_dist} length")
        else:
            fail_list.append(question_id)
            logging.info(f"Question id {question_id} failed, {explore_dist} length")
        logging.info(f"{question_idx + 1}/{total_questions}: Success rate: {success_count}/{question_idx + 1}")
        logging.info(f"Mean path length for success exploration: {np.mean(list(path_length_list.values()))}")
        logging.info(f"Filtered snapshots/Total snapshots/Total frames: {n_filtered_snapshots}/{n_total_snapshots}/{n_total_frames}")

        # save the gpt answer
        if gpt_answer is not None:
            gpt_answer_list.append({
                "question_id": question_id,
                "answer": gpt_answer
            })

        # save the number of snapshots
        n_filtered_snapshots_list[question_id] = n_filtered_snapshots
        n_total_snapshots_list[question_id] = n_total_snapshots
        n_total_frames_list[question_id] = n_total_frames

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
                    step_dict["snapshot_objects"] = {}
                    step_dict["snapshot_imgs"] = {}
                    snapshot_id_mapping = {}
                    prompt_ss_idx = 0
                    for snapshot_idx, (rgb_id, snapshot) in enumerate(scene.snapshots.items()):
                        if rgb_id in selected_snapshot_ids:
                            continue
                        step_dict["snapshot_objects"][rgb_id] = snapshot.cluster
                        step_dict["snapshot_imgs"] = all_snapshots[rgb_id]

                        snapshot_id_mapping[prompt_ss_idx] = snapshot_idx
                        prompt_ss_idx += 1

                    if len(snapshot_id_mapping) == 0:
                        logging.info(f"Question id {question_id} target not found handling: no snapshots available!")
                        break

                    # add scene graph
                    step_dict["obj_map"] = object_id_to_name

                    # we don't need to add frontier for model selection
                    step_dict["frontiers"] = []

                    if cfg.egocentric_views:
                        step_dict["use_egocentric_views"] = True
                        step_dict["egocentric_imgs"] = rgb_egocentric_views

                    step_dict["frontier_imgs"] = []
                    step_dict["question"] = question
                    step_dict["scene"] = scene_id

                    outputs, snapshot_id_mapping = explore_step(step_dict, cfg)
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
            else:
                raise ValueError(f"Invalid target not found handling method: {cfg.handle_target_not_found}")

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
        logging.info(f"Prediction: {gpt_answer}")
        for snapshot_id, obj_list in snapshot_dict.items():
            logging.info(f"{snapshot_id}:")
            for obj_str in obj_list:
                logging.info(f"\t{obj_str}")

        with open(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)
        with open(os.path.join(str(cfg.output_dir), f"fail_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
            pickle.dump(fail_list, f)
        with open(os.path.join(str(cfg.output_dir), f"gpt_answer_{start_ratio}_{end_ratio}.json"), "w") as f:
            json.dump(gpt_answer_list, f, indent=4)
        with open(os.path.join(str(cfg.output_dir), f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"), "w") as f:
            json.dump(n_filtered_snapshots_list, f, indent=4)
        with open(os.path.join(str(cfg.output_dir), f"n_total_snapshots_{start_ratio}_{end_ratio}.json"), "w") as f:
            json.dump(n_total_snapshots_list, f, indent=4)
        with open(os.path.join(str(cfg.output_dir), f"n_total_frames_{start_ratio}_{end_ratio}.json"), "w") as f:
            json.dump(n_total_frames_list, f, indent=4)

        # clear up memory
        if not cfg.save_visualization:
            os.system(f"rm -r {episode_data_dir}")

    with open(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)
    with open(os.path.join(str(cfg.output_dir), f"fail_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(fail_list, f)
    with open(os.path.join(str(cfg.output_dir), f"gpt_answer_{start_ratio}_{end_ratio}.json"), "w") as f:
        json.dump(gpt_answer_list, f, indent=4)
    with open(os.path.join(str(cfg.output_dir), f"n_filtered_snapshots_{start_ratio}_{end_ratio}.json"), "w") as f:
        json.dump(n_filtered_snapshots_list, f, indent=4)
    with open(os.path.join(str(cfg.output_dir), f"n_total_snapshots_{start_ratio}_{end_ratio}.json"), "w") as f:
        json.dump(n_total_snapshots_list, f, indent=4)
    with open(os.path.join(str(cfg.output_dir), f"n_total_frames_{start_ratio}_{end_ratio}.json"), "w") as f:
        json.dump(n_total_frames_list, f, indent=4)

    logging.info(f'All scenes finish')

    # aggregate the results into a single file
    success_list = []
    path_length_list = {}
    all_success_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "success_list_*.pkl"))
    all_path_length_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "path_length_list_*.pkl"))
    for success_list_path in all_success_list_paths:
        with open(success_list_path, "rb") as f:
            success_list += pickle.load(f)
    for path_length_list_path in all_path_length_list_paths:
        with open(path_length_list_path, "rb") as f:
            path_length_list.update(pickle.load(f))

    with open(os.path.join(str(cfg.output_dir), "success_list.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), "path_length_list.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)

    gpt_answer_list = []
    all_gpt_answer_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "gpt_answer_*.json"))
    for gpt_answer_list_path in all_gpt_answer_list_paths:
        with open(gpt_answer_list_path, "r") as f:
            gpt_answer_list += json.load(f)

    with open(os.path.join(str(cfg.output_dir), "gpt_answer.json"), "w") as f:
        json.dump(gpt_answer_list, f, indent=4)

    n_filtered_snapshots_list = {}
    all_n_filtered_snapshots_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "n_filtered_snapshots_*.json"))
    for n_filtered_snapshots_list_path in all_n_filtered_snapshots_list_paths:
        with open(n_filtered_snapshots_list_path, "r") as f:
            n_filtered_snapshots_list.update(json.load(f))

    with open(os.path.join(str(cfg.output_dir), "n_filtered_snapshots.json"), "w") as f:
        json.dump(n_filtered_snapshots_list, f, indent=4)
    logging.info(f"Average number of filtered snapshots: {np.mean(list(n_filtered_snapshots_list.values()))}")

    n_total_snapshots_list = {}
    all_n_total_snapshots_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "n_total_snapshots_*.json"))
    for n_total_snapshots_list_path in all_n_total_snapshots_list_paths:
        with open(n_total_snapshots_list_path, "r") as f:
            n_total_snapshots_list.update(json.load(f))

    with open(os.path.join(str(cfg.output_dir), "n_total_snapshots.json"), "w") as f:
        json.dump(n_total_snapshots_list, f, indent=4)
    logging.info(f"Average number of total snapshots: {np.mean(list(n_total_snapshots_list.values()))}")

    n_total_frames_list = {}
    all_n_total_frames_list_paths = glob.glob(os.path.join(str(cfg.output_dir), "n_total_frames_*.json"))
    for n_total_frames_list_path in all_n_total_frames_list_paths:
        with open(n_total_frames_list_path, "r") as f:
            n_total_frames_list.update(json.load(f))

    with open(os.path.join(str(cfg.output_dir), "n_total_frames.json"), "w") as f:
        json.dump(n_total_frames_list, f, indent=4)
    logging.info(f"Average number of total frames: {np.mean(list(n_total_frames_list.values()))}")


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
    main(cfg, args.start_ratio, args.end_ratio)
