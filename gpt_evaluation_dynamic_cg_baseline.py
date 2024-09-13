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
from src.scene_cg_baseline import Scene
from src.eval_utils_snapshot_new import rgba2rgb
from src.eval_utils_gpt_cg_baseline import explore_step


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

    success_count = 0
    max_target_observation = cfg.max_target_observation

    # Run all questions
    for question_idx, question_data in enumerate(questions_list):
        question_id = question_data['question_id']
        question = question_data['question']
        answer = question_data['answer']

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

        if len(os.listdir(episode_object_observe_dir)) > 0:
            logging.info(f"Question id {question_id} already has enough target observations!")
            success_count += 1
            continue

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
        target_observation_count = 0

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
                obs, cam_pose = scene.get_observation(pts, ang)
                rgb = obs["color_sensor"]
                depth = obs["depth_sensor"]
                semantic_obs = obs["semantic_sensor"]
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
                    rgb_egocentric_views.append(resized_rgb)
                    if cfg.save_visualization or cfg.save_frontier_video:
                        plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), annotated_rgb)
                    else:
                        plt.imsave(os.path.join(episode_snapshot_dir, obs_file_name), rgb)

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

            logging.info(f"Step {cnt_step} {len(scene.objects)} objects")

            # update the mapping of object id to class name, since the objects have been updated
            object_id_to_name = {obj_id: obj["class_name"] for obj_id, obj in scene.objects.items()}
            step_dict["obj_map"] = object_id_to_name

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

                step_dict["objects"] = {}
                for obj_id, obj in scene.objects.items():
                    step_dict["objects"][obj_id] = obj["image_crop"]  # obj_id -> image_crop: pil_image

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

                outputs, obj_id_mapping = explore_step(step_dict, cfg)
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

                if target_type not in ["object", "frontier"]:
                    logging.info(f"Invalid prediction type: {target_type}, failed!")
                    print(target_type)
                    break

                if target_type == "object":
                    # TODO: the problem needed to be fixed here
                    if obj_id_mapping is not None:
                        if int(target_index) < 0 or int(target_index) >= len(obj_id_mapping):
                            logging.info(f"target index can not match real objects: {target_index}, failed!")
                            break
                        target_index = obj_id_mapping[int(target_index)]
                        logging.info(f"The index of target object {target_index}")
                    if int(target_index) < 0 or int(target_index) >= len(scene.objects):
                        logging.info(f"Prediction out of range: {target_index}, {len(scene.objects)}, failed!")
                        break
                    pred_target_object = list(scene.objects.values())[int(target_index)]
                    logging.info(f"Predicted object: id: {pred_target_object['id']}, class: {pred_target_object['class_name']}, num_detections: {pred_target_object['num_detections']}")

                    # construct a snapshot for the object
                    pred_target_snapshot = SnapShot(
                        image="no_use",
                        color=(random.random(), random.random(), random.random()),
                        obs_point=np.empty(3),
                        full_obj_list={pred_target_object['id']: pred_target_object['conf']},
                        cluster=[pred_target_object['id']],
                    )

                    tsdf_planner.frontiers_weight = np.zeros((len(tsdf_planner.frontiers)))
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
                snapshots={},
                pathfinder=scene.pathfinder,
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
                            obj_id = max_point_choice.cluster[0]
                            img_pil = scene.objects[obj_id]["image_crop"]
                            img = np.array(img_pil)

                            axs[h_idx, w_idx].imshow(img)
                            axs[h_idx, w_idx].set_title('Object Chosen')
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

            if type(max_point_choice) == SnapShot and target_arrived:
                # get an observation and break
                obs, _ = scene.get_observation(pts, angle)
                rgb = obs["color_sensor"]

                plt.imsave(
                    os.path.join(episode_object_observe_dir, f"target_{target_observation_count}.png"), rgb
                )
                # also, save the image crop of the object in the snapshot
                obj_id = max_point_choice.cluster[0]
                img_pil = scene.objects[obj_id]["image_crop"]
                img = np.array(img_pil)
                plt.imsave(
                    os.path.join(episode_object_observe_dir, f"target_{target_observation_count}_crop.png"), img
                )

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

        logging.info(f"Scene graph of question {question_id}:")
        logging.info(f"Question: {question}")
        logging.info(f"Answer: {answer}")

        with open(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
            pickle.dump(success_list, f)
        with open(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
            pickle.dump(path_length_list, f)

    with open(os.path.join(str(cfg.output_dir), f"success_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(success_list, f)
    with open(os.path.join(str(cfg.output_dir), f"path_length_list_{start_ratio}_{end_ratio}.pkl"), "wb") as f:
        pickle.dump(path_length_list, f)

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
