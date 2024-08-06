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

np.set_printoptions(precision=3)
import json
import logging
import glob
from habitat_sim.utils.common import quat_from_two_vectors, quat_to_angle_axis
import open_clip
from ultralytics import YOLO, SAM
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
from inference.models import YOLOWorld

from conceptgraph.utils.general_utils import measure_time


def main(cfg):
    # use hydra to load concept graph related configs
    # @hydra.main(version_base=None, config_path=cfg.concept_graph_config_path, config_name=cfg.concept_graph_config_name)
    # def get_conceptgraph_config(conf):
    #     conf = process_cfg(conf)
    #     return conf
    # cfg_cg = get_conceptgraph_config()

    with initialize(config_path="conceptgraph/hydra_configs", job_name="app"):
        cfg_cg = compose(config_name=cfg.concept_graph_config_name)

    img_height = cfg.img_height
    img_width = cfg.img_width
    cam_intr = get_cam_intr(cfg.hfov, img_height, img_width)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # load object detection model
    detection_model_yoloworld = YOLOWorld(model_id=cfg.detection_model_name)

    # Load dataset
    with open(os.path.join(cfg.question_data_path, "generated_questions.json")) as f:
        questions_data = json.load(f)
    questions_data = sorted(questions_data, key=lambda x: x["episode_history"])
    questions_data = questions_data[int(args.start * len(questions_data)):int(args.end * len(questions_data))]
    all_scene_list = list(set([q["episode_history"] for q in questions_data]))
    logging.info(f"Loaded {len(questions_data)} questions.")

    ## Initialize the detection models
    detection_model = measure_time(YOLO)('yolov8x-world.pt')   # yolov8x-world.pt
    sam_predictor = SAM('sam_l.pt')  # SAM('sam_l.pt') # UltraLytics SAM
    # sam_predictor = measure_time(get_sam_predictor)(cfg) # Normal SAM
    clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", "laion2b_s34b_b79k"  # "ViT-H-14", "laion2b_s32b_b79k"
    )
    clip_model = clip_model.to(cfg_cg.device)
    clip_tokenizer = open_clip.get_tokenizer("ViT-B-32")

    total_images_record = []

    # for each scene, answer each question
    question_ind = 0
    success_count = 0
    total_questions = len(questions_data) * cfg.paths_per_question
    for scene_id in all_scene_list:
        all_questions_in_scene = [q for q in questions_data if q["episode_history"] == scene_id]

        ##########################################################
        if '00538' not in scene_id:
            continue
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
        try:
            del scene
        except:
            pass

        scene = Scene(scene_id, cfg, cfg_cg)
        # Set the classes for the detection model
        detection_model.set_classes(scene.obj_classes.get_classes_arr())

        for question_data in all_questions_in_scene:
            # for each question, generate several paths, starting from different starting points
            for path_idx in range(cfg.path_id_offset, cfg.path_id_offset + cfg.paths_per_question):
                question_ind += 1

                target_obj_id = question_data['object_id']
                target_obj_name = question_data['class']
                target_position = question_data['position']
                target_rotation = question_data['rotation']
                episode_data_dir = os.path.join(str(cfg.dataset_output_dir), f"{question_data['question_id']}_path_{path_idx}")
                episode_frontier_dir = os.path.join(episode_data_dir, "frontier_rgb")
                egocentric_save_dir = os.path.join(episode_data_dir, 'egocentric')
                object_feature_save_dir = os.path.join(episode_data_dir, 'object_features')

                # if the data has already generated, skip
                if os.path.exists(episode_data_dir) and os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} already exists")
                    success_count += 1
                    continue

                if os.path.exists(episode_data_dir) and not os.path.exists(os.path.join(episode_data_dir, "metadata.json")):
                    os.system(f"rm -r {episode_data_dir}")

                os.makedirs(episode_data_dir, exist_ok=True)
                os.makedirs(episode_frontier_dir, exist_ok=True)
                os.makedirs(egocentric_save_dir, exist_ok=True)
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
                tsdf_bnds, scene_size = get_scene_bnds(scene.pathfinder, floor_height)
                scene_length = (tsdf_bnds[:2, 1] - tsdf_bnds[:1, 0]).mean()
                min_dist = cfg.min_travel_dist
                start_position, path_points, travel_dist = scene.get_navigable_point_to(
                    target_position, max_search=1000, min_dist=min_dist, max_dist=min_dist + 5,
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

                # clear up the previous detected frontiers and objects
                scene.clear_up_detections()

                logging.info(f'\n\nQuestion id {question_data["question_id"]}-path {path_idx} initialization successful!')

                metadata = {}
                metadata["question"] = question_data["question"]
                metadata["answer"] = question_data["answer"]
                metadata["scene"] = question_data["episode_history"]
                metadata["init_pts"] = pts.tolist()
                metadata["init_angle"] = angle
                metadata["target_obj_id"] = target_obj_id
                metadata["target_obj_class"] = question_data["class"]

                # run steps
                target_found = False
                previous_choice_path = None
                max_explore_dist = travel_dist * cfg.max_step_dist_ratio
                max_step = int(travel_dist * cfg.max_step_ratio)
                target_obj_id_det_list = []  # record all the detected target object ids, and use the most frequent one as the final target object id
                explore_dist = 0.0
                cnt_step = -1
                while explore_dist < max_explore_dist and cnt_step < max_step:
                    cnt_step += 1

                    step_dict = {}
                    step_dict["agent_state"] = {}
                    step_dict["agent_state"]["init_pts"] = pts.tolist()
                    step_dict["agent_state"]["init_angle"] = rotation

                    logging.info(f"\n== step: {cnt_step}")

                    # for each position, get the views from different angles
                    if len(target_obj_id_det_list) > 0:
                        # if the target object is detected
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
                    for view_idx, ang in enumerate(all_angles):
                        obs, cam_pose = scene.get_observation(pts, ang)
                        rgb = obs["color_sensor"]
                        depth = obs["depth_sensor"]
                        semantic_obs = obs["semantic_sensor"]

                        cam_pose_normal = pose_habitat_to_normal(cam_pose)
                        cam_pose_tsdf = pose_normal_to_tsdf(cam_pose_normal)

                        obs_file_name = f"{cnt_step}-view_{view_idx}.png"
                        annotated_rgb, object_added, target_obj_id_det = scene.update_scene_graph(
                            image_rgb=rgb[..., :3], depth=depth, intrinsics=cam_intr, cam_pos=cam_pose,
                            detection_model=detection_model, sam_predictor=sam_predictor, clip_model=clip_model,
                            clip_preprocess=clip_preprocess, clip_tokenizer=clip_tokenizer,
                            pts=pts, pts_voxel=tsdf_planner.habitat2voxel(pts),
                            img_path=obs_file_name,
                            frame_idx=cnt_step * total_views + view_idx,
                            target_obj_mask=semantic_obs == target_obj_id,
                        )
                        if object_added:
                            plt.imsave(os.path.join(object_feature_save_dir, obs_file_name), annotated_rgb)
                        if target_obj_id_det is not None:
                            target_obj_id_det_list.append(target_obj_id_det)

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

                        if cfg.save_egocentric_view:
                            plt.imsave(os.path.join(egocentric_save_dir, f"{cnt_step}_view_{view_idx}.png"), rgb)

                    scene.update_snapshots(min_num_obj_threshold=cfg.min_num_obj_threshold)
                    logging.info(f"Step {cnt_step} {len(scene.objects)} objects, {len(scene.snapshots)} snapshots")

                    update_success = tsdf_planner.update_frontier_map(pts=pts_normal, cfg=cfg.planner)
                    if not update_success:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: update frontier map failed!")
                        break

                    if target_found:
                        break

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

                            obs, target_detected = scene.get_frontier_observation_and_detect_target(
                                pts, view_frontier_direction, detection_model_yoloworld, target_obj_id, target_obj_name
                            )
                            frontier_obs = obs["color_sensor"]
                            if target_detected:
                                frontier.target_detected = True

                            plt.imsave(
                                os.path.join(episode_frontier_dir, f"{cnt_step}_{i}.png"),
                                frontier_obs,
                            )
                            frontier.image = f"{cnt_step}_{i}.png"
                            frontier_dict["rgb_id"] = f"{cnt_step}_{i}.png"
                        step_dict["frontiers"].append(frontier_dict)

                    # reset target point to allow the model to choose again
                    tsdf_planner.max_point = None
                    tsdf_planner.target_point = None

                    print(f'!!!!!!! {target_obj_id_det_list}')

                    # use the most common id in the list as the target object id
                    if len(target_obj_id_det_list) > 0:
                        target_obj_id_det = max(set(target_obj_id_det_list), key=target_obj_id_det_list.count)
                        logging.info(f"Target object {target_obj_id_det} {scene.objects[target_obj_id_det]['class_name']} used for selecting snapshot!")
                    else:
                        target_obj_id_det = -1

                    max_point_choice = tsdf_planner.get_next_choice(
                        pts=pts_normal,
                        objects=scene.objects,
                        snapshots=scene.snapshots,
                        path_points=path_points,
                        pathfinder=scene.pathfinder,
                        target_obj_id=target_obj_id_det,
                        cfg=cfg.planner,
                    )
                    if max_point_choice is None:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: no valid choice!")
                        break

                    update_success = tsdf_planner.set_next_navigation_point(
                        choice=max_point_choice,
                        pts=pts_normal,
                        objects=scene.objects,
                        cfg=cfg.planner,
                        pathfinder=scene.pathfinder,
                    )
                    if not update_success:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: set_next_navigation_point failed!")
                        break

                    return_values = tsdf_planner.agent_step(
                        pts=pts_normal,
                        angle=angle,
                        objects=scene.objects,
                        snapshots=scene.snapshots,
                        pathfinder=scene.pathfinder,
                        cfg=cfg.planner,
                        path_points=path_points,
                        save_visualization=cfg.save_visualization,
                    )
                    if return_values[0] is None:
                        logging.info(f"Question id {question_data['question_id']}-path {path_idx} invalid: find next navigation point failed!")
                        break
                    pts_normal, angle, pts_pix, fig, path_points, target_arrived = return_values

                    # save snapshots
                    step_dict["snapshots"] = []
                    for snapshot in scene.snapshots.values():
                        step_dict["snapshots"].append(
                            {
                                "img_id": snapshot.image,
                                "obj_ids": [int(obj_id) for obj_id in snapshot.selected_obj_list]
                             }
                        )

                    # tempt
                    step_dict["scene_graph_file2objs"] = {}
                    for obj_id, obj in scene.objects.items():
                        if obj['image'] not in step_dict["scene_graph_file2objs"]:
                            step_dict["scene_graph_file2objs"][obj['image']] = []
                        step_dict["scene_graph_file2objs"][obj['image']].append(
                            f"{obj_id}: {obj['class_name']}"
                        )

                    # sanity check
                    assert len(step_dict["snapshots"]) == len(scene.snapshots), f"{len(step_dict['snapshots'])} != {len(scene.snapshots)}"
                    total_objs_count = 0
                    for snapshot in scene.snapshots.values():
                        total_objs_count += len(snapshot.selected_obj_list)
                    assert len(scene.objects) == total_objs_count, f"{len(scene.objects)} != {total_objs_count}"
                    for obj_id in scene.objects.keys():
                        exist_count = 0
                        for ss in scene.snapshots.values():
                            if obj_id in ss.selected_obj_list:
                                exist_count += 1
                        assert exist_count == 1, f"{exist_count} != 1 for obj_id {obj_id}, {scene.objects[obj_id]['class_name']}"
                    for ss in scene.snapshots.values():
                        assert len(ss.selected_obj_list) == len(set(ss.selected_obj_list)), f"{ss.selected_obj_list} has duplicates"
                        assert len(ss.full_obj_list.keys()) == len(set(ss.full_obj_list.keys())), f"{ss.full_obj_list.keys()} has duplicates"
                        for obj_id in ss.selected_obj_list:
                            assert obj_id in ss.full_obj_list, f"{obj_id} not in {ss.full_obj_list.keys()}"
                        for obj_id in ss.full_obj_list.keys():
                            assert obj_id in scene.objects, f"{obj_id} not in scene objects"

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

                        # # save ax1 as another image
                        # ax1 = fig.axes[0]
                        # extent = ax1.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
                        # fig.savefig(os.path.join(visualization_path, f"objects_{cnt_step}.png"), bbox_inches=extent)

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
                        obs, _ = scene.get_observation(pts, angle)
                        rgb = obs["color_sensor"]

                        target_obs_save_dir = os.path.join(episode_data_dir, "target_observation")
                        os.makedirs(target_obs_save_dir, exist_ok=True)
                        plt.imsave(os.path.join(target_obs_save_dir, f"{cnt_step}_target_observation.png"), rgb)
                        break

                if target_found:
                    metadata["episode_length"] = cnt_step
                    with open(os.path.join(episode_data_dir, "metadata.json"), "w") as f:
                        json.dump(metadata, f, indent=4)
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} finish with {cnt_step} steps")
                    success_count += 1
                else:
                    logging.info(f"Question id {question_data['question_id']}-path {path_idx} failed.")
                    if cfg.del_fail_case:
                        os.system(f"rm -r {episode_data_dir}")

                logging.info(f"{question_ind}/{total_questions}: Success rate: {success_count}/{question_ind}")







                # print the stats of total number of images
                img_dict = {}
                for obj_id, obj in scene.objects.items():
                    if obj['image'] not in img_dict:
                        img_dict[obj['image']] = []
                    img_dict[obj['image']].append(obj['id'])
                total_images = len(img_dict.keys())
                total_images_record.append(total_images)
                logging.info('\n')
                logging.info(total_images_record)
                logging.info(f"{question_data['question_id']}-path {path_idx} total images: {total_images}")
                logging.info(f"Average total images: {np.mean(total_images_record):.2f} +- {np.std(total_images_record):.2f}")
                logging.info(f"Max total images: {np.max(total_images_record)}, Min total images: {np.min(total_images_record)}")






            logging.info(f"Question id {question_data['question_id']} finished all paths")

        logging.info(f'Scene {scene_id} finish')

    logging.info(f'All scenes finish')


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
    logging_path = os.path.join(cfg.output_dir, "log.log")
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
