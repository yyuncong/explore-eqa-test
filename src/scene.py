import os
import numpy as np
import logging
import random
import torch
import habitat_sim
import quaternion
from quaternion import as_float_array
import supervision as sv
import logging
from collections import Counter
from scipy.spatial.transform import Rotation
from typing import List, Optional, Tuple, Dict, Union
from dataclasses import dataclass, field

from habitat_sim.utils.common import quat_to_coeffs, quat_from_angle_axis, quat_from_two_vectors
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
from src.geom import get_cam_intr, IoU
from src.tsdf_new_cg import SnapShot

# Local application/library specific imports
from conceptgraph.utils.optional_rerun_wrapper import (
    OptionalReRun,
    orr_log_annotated_image,
    orr_log_camera,
    orr_log_depth_image,
    orr_log_edges,
    orr_log_objs_pcd_and_bbox,
    orr_log_rgb_image,
    orr_log_vlm_image
)
from conceptgraph.utils.optional_wandb_wrapper import OptionalWandB
from conceptgraph.utils.geometry import rotation_matrix_to_quaternion
from conceptgraph.utils.logging_metrics import DenoisingTracker, MappingTracker
from conceptgraph.utils.vlm import get_obj_rel_from_image_gpt4v, get_openai_client
from conceptgraph.utils.ious import mask_subtract_contained
from conceptgraph.utils.general_utils import (
    ObjectClasses,
    find_existing_image_path,
    get_det_out_path,
    get_exp_out_path,
    get_stream_data_out_path,
    get_vlm_annotated_image_path,
    handle_rerun_saving,
    load_saved_detections,
    load_saved_hydra_json_config,
    make_vlm_edges_and_captions,
    measure_time,
    save_detection_results,
    save_hydra_config,
    save_objects_for_frame,
    save_pointcloud,
    should_exit_early,
    vis_render_image,
    filter_detections
)
from conceptgraph.dataset.datasets_common import get_dataset
from conceptgraph.utils.vis import (
    OnlineObjectRenderer,
    save_video_from_frames,
    vis_result_fast_on_depth,
    vis_result_for_vlm,
    vis_result_fast,
    save_video_detections
)
from conceptgraph.slam.slam_classes import MapObjectDict, DetectionDict, to_tensor
from conceptgraph.slam.utils import (
    filter_gobs,
    filter_objects,
    get_bounding_box,
    init_process_pcd,
    make_detection_list_from_pcd_and_gobs,
    denoise_objects,
    merge_objects,
    detections_to_obj_pcd_and_bbox,
    prepare_objects_save_vis,
    process_cfg,
    process_edges,
    process_pcd,
    processing_needed,
    resize_gobs,
    merge_obj2_into_obj1
)
from conceptgraph.slam.mapping import (
    compute_spatial_similarities,
    compute_visual_similarities,
    aggregate_similarities,
    match_detections_to_objects,
)
from conceptgraph.utils.model_utils import compute_clip_features_batched
from conceptgraph.utils.general_utils import get_vis_out_path, cfg_to_dict, check_run_detections


class Scene:
    def __init__(
            self,
            scene_id,
            cfg,
            graph_cfg,
    ):
        self.cfg = cfg
        self.cfg_cg = graph_cfg

        # about the loading the scene
        split = "train" if int(scene_id.split("-")[0]) < 800 else "val"
        scene_mesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(cfg.scene_data_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        assert os.path.exists(scene_mesh_path), f"scene_mesh_path: {scene_mesh_path} does not exist"
        assert os.path.exists(navmesh_path), f"navmesh_path: {navmesh_path} does not exist"
        assert os.path.exists(semantic_texture_path), f"semantic_texture_path: {semantic_texture_path} does not exist"
        assert os.path.exists(scene_semantic_annotation_path), f"scene_semantic_annotation_path: {scene_semantic_annotation_path} does not exist"

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": cfg.camera_height,
            "width": cfg.img_width,
            "height": cfg.img_height,
            "hfov": cfg.hfov,
            "scene_dataset_config_file": cfg.scene_dataset_config_path,
            "camera_tilt": cfg.camera_tilt_deg * np.pi / 180,
        }
        sim_cfg = make_semantic_cfg_new(sim_settings)
        self.simulator = habitat_sim.Simulator(sim_cfg)
        self.pathfinder = self.simulator.pathfinder
        self.pathfinder.seed(cfg.seed)
        self.pathfinder.load_nav_mesh(navmesh_path)

        # load object classes
        self.obj_classes = ObjectClasses(
            classes_file_path=scene_semantic_annotation_path,
            bg_classes=self.cfg_cg['bg_classes'],
            skip_bg=self.cfg_cg['skip_bg']
        )

        logging.info(f"Load scene {scene_id} successfully")

        # set agent
        self.agent = self.simulator.initialize_agent(sim_settings["default_agent"])

        self.cam_intrinsic = get_cam_intr(cfg.img_width, cfg.img_height, cfg.hfov)

        # about scene graph
        self.objects: MapObjectDict[int, Dict] = MapObjectDict()  # object_id -> object item
        self.object_id_counter = 1

        self.snapshots: Dict[str, SnapShot] = {}  # image_path -> snapshot


    def __del__(self):
        try:
            self.simulator.close()
        except:
            pass

    def clear_up_detections(self):
        self.objects = MapObjectDict()
        self.object_id_counter = 1
        self.snapshots = {}

    def get_observation(self, pts, angle):
        agent_state = habitat_sim.AgentState()
        agent_state.position = pts
        agent_state.rotation = get_quaternion(angle, 0)
        self.agent.set_state(agent_state)

        obs = self.simulator.get_sensor_observations()

        # get camera extrinsic matrix
        sensor = self.agent.get_state().sensor_states["depth_sensor"]
        quaternion_0 = sensor.rotation
        translation_0 = sensor.position
        cam_pose = np.eye(4)
        cam_pose[:3, :3] = quaternion.as_rotation_matrix(quaternion_0)
        cam_pose[:3, 3] = translation_0

        return obs, cam_pose

    def get_frontier_observation(self, pts, view_dir, camera_tilt=0.0):
        agent_state = habitat_sim.AgentState()

        # solve edge cases of viewing direction
        default_view_dir = np.asarray([0., 0., -1.])
        if np.linalg.norm(view_dir) < 1e-3:
            view_dir = default_view_dir
        view_dir = view_dir / np.linalg.norm(view_dir)

        agent_state.position = pts
        # set agent observation direction
        if np.dot(view_dir, default_view_dir) / np.linalg.norm(view_dir) < -1 + 1e-3:
            # if the rotation is to rotate 180 degree, then the quaternion is not unique
            # we need to specify rotating along y-axis
            agent_state.rotation = quat_to_coeffs(
                quaternion.quaternion(0, 0, 1, 0)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()
        else:
            agent_state.rotation = quat_to_coeffs(
                quat_from_two_vectors(default_view_dir, view_dir)
                * quat_from_angle_axis(camera_tilt, np.array([1, 0, 0]))
            ).tolist()

        self.agent.set_state(agent_state)
        obs = self.simulator.get_sensor_observations()

        return obs

    def get_frontier_observation_and_detect_target(
            self,
            pts, view_dir,
            detection_model, target_obj_id, target_obj_class,
            camera_tilt=0.0
    ):
        obs = self.get_frontier_observation(pts, view_dir, camera_tilt)

        # detect target object
        rgb = obs["color_sensor"]
        semantic_obs = obs["semantic_sensor"]

        detection_model.set_classes([target_obj_class])
        results = detection_model.infer(rgb[..., :3], confidence=self.cfg.scene_graph.confidence)
        detections = sv.Detections.from_inference(results).with_nms(threshold=self.cfg.scene_graph.nms_threshold)

        target_detected = False
        if target_obj_id in np.unique(semantic_obs):
            for i in range(len(detections)):
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                bbox_mask = np.zeros(semantic_obs.shape, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True

                target_x_start, target_y_start = np.argwhere(semantic_obs == target_obj_id).min(axis=0)
                target_x_end, target_y_end = np.argwhere(semantic_obs == target_obj_id).max(axis=0)
                obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
                obj_mask[target_x_start:target_x_end, target_y_start:target_y_end] = True
                if IoU(bbox_mask, obj_mask) > self.cfg.scene_graph.iou_threshold:
                    target_detected = True
                    break

        return obs, target_detected

    def get_navigable_point_to(
            self,
            target_position, max_search=1000, min_dist=6.0, max_dist=999.0,
            prev_start_positions=None
    ):
        self.pathfinder.seed(random.randint(0, 1000000))
        return get_navigable_point_to_new(
            target_position, self.pathfinder, max_search, min_dist, max_dist, prev_start_positions
        )

    def update_scene_graph(
            self,
            image_rgb: np.ndarray, depth: np.ndarray, intrinsics, cam_pos,
            detection_model, sam_predictor, clip_model, clip_preprocess, clip_tokenizer,
            pts, pts_voxel,
            img_path, frame_idx,
            target_obj_mask=None  # the boolean mask of target object generated from the semantic sensor. If given, return the object id of the target object
    ) -> Tuple[np.ndarray, bool, Optional[int]]:
        # return annotated image; whether any object is added to scene graph in this frame; the object id of the target object (if detected)

        obj_classes = self.obj_classes
        # Detect objects
        results = detection_model.predict(image_rgb, conf=0.1, verbose=False)
        confidences = results[0].boxes.conf.cpu().numpy()
        detection_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
        detection_class_labels = [f"{obj_classes.get_classes_arr()[class_id]} {class_idx}" for class_idx, class_id in enumerate(detection_class_ids)]
        xyxy_tensor = results[0].boxes.xyxy
        xyxy_np = xyxy_tensor.cpu().numpy()

        # if there are detections,
        # Get Masks Using SAM or MobileSAM
        # UltraLytics SAM
        if xyxy_tensor.numel() != 0:
            sam_out = sam_predictor.predict(image_rgb, bboxes=xyxy_tensor, verbose=False)
            masks_tensor = sam_out[0].masks.data

            masks_np = masks_tensor.cpu().numpy()
        else:
            masks_np = np.empty((0, *image_rgb.shape[:2]), dtype=np.float64)

        # Create a detections object that we will save later
        curr_det = sv.Detections(
            xyxy=xyxy_np,
            confidence=confidences,
            class_id=detection_class_ids,
            mask=masks_np,
        )

        if len(curr_det) == 0:  # no detections, skip
            logging.debug("No detections in this frame")
            return image_rgb, False, None

        # filter the detection by removing overlapping detections
        curr_det, labels = filter_detections(
            image=image_rgb,
            detections=curr_det,
            classes=obj_classes,
            given_labels=detection_class_labels,
            iou_threshold=self.cfg_cg.object_detection_iou_threshold,
            min_mask_size_ratio=self.cfg_cg.min_mask_size_ratio,
            confidence_threshold=self.cfg_cg.object_detection_confidence_threshold,
        )
        if curr_det is None:
            logging.debug("No detections left after filter_detections")
            return image_rgb, False, None

        image_crops, image_feats, text_feats = compute_clip_features_batched(
            image_rgb, curr_det, clip_model, clip_preprocess, clip_tokenizer, obj_classes.get_classes_arr(), self.cfg_cg.device)

        raw_gobs = {
                # add new uuid for each detection
                "xyxy": curr_det.xyxy,
                "confidence": curr_det.confidence,
                "class_id": curr_det.class_id,
                "mask": curr_det.mask,
                "classes": obj_classes.get_classes_arr(),
                "image_crops": image_crops,
                "image_feats": image_feats,
                "text_feats": text_feats,
                "detection_class_labels": detection_class_labels,
            }

        # resize the observation if needed
        resized_gobs = resize_gobs(raw_gobs, image_rgb)
        # filter the observations
        filtered_gobs = filter_gobs(
            resized_gobs, image_rgb,
            skip_bg=self.cfg_cg.skip_bg,
            BG_CLASSES=obj_classes.get_bg_classes_arr(),
            mask_area_threshold=self.cfg_cg.mask_area_threshold,
            max_bbox_area_ratio=self.cfg_cg.max_bbox_area_ratio,
            mask_conf_threshold=self.cfg_cg.mask_conf_threshold,
        )

        gobs = filtered_gobs

        if len(gobs['mask']) == 0: # no detections in this frame
            logging.debug("No detections left after filter_gobs")
            return image_rgb, False, None

        # this helps make sure things like pillows on couches are separate objects
        gobs['mask'] = mask_subtract_contained(gobs['xyxy'], gobs['mask'])

        obj_pcds_and_bboxes = measure_time(detections_to_obj_pcd_and_bbox)(
            depth_array=depth,
            masks=gobs['mask'],
            cam_K=intrinsics[:3, :3],  # Camera intrinsics
            image_rgb=image_rgb,
            trans_pose=cam_pos,
            min_points_threshold=self.cfg_cg.min_points_threshold,
            spatial_sim_type=self.cfg_cg.spatial_sim_type,
            obj_pcd_max_points=self.cfg_cg.obj_pcd_max_points,
            device=self.cfg_cg.device,
        )

        for obj in obj_pcds_and_bboxes:
            if obj:
                obj["pcd"] = init_process_pcd(
                    pcd=obj["pcd"],
                    downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                    dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                    dbscan_eps=self.cfg_cg["dbscan_eps"],
                    dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                )
                obj["bbox"] = get_bounding_box(
                    spatial_sim_type=self.cfg_cg['spatial_sim_type'],
                    pcd=obj["pcd"],
                )
        # if the list is all None, then skip
        if all([obj is None for obj in obj_pcds_and_bboxes]):
            logging.debug("All objects are None in obj_pcds_and_bboxes")
            return image_rgb, False, None

        # add pcds and bboxes to gobs
        gobs['bbox'] = [obj["bbox"] if obj is not None else None for obj in obj_pcds_and_bboxes]
        gobs['pcd'] = [obj["pcd"] if obj is not None else None for obj in obj_pcds_and_bboxes]

        # filter out objects that are far away
        gobs = self.filter_gobs_with_distance(pts, gobs)

        detection_list = self.make_detection_list_from_pcd_and_gobs(
            gobs, img_path, obj_classes
        )

        if len(detection_list) == 0:  # no detections, skip
            logging.debug("No detections left after make_detection_list_from_pcd_and_gobs")
            return image_rgb, False, None

        # compare the detections with the target object mask to see whether the target object is detected
        target_obj_id = None
        if target_obj_mask is not None and np.sum(target_obj_mask) / (target_obj_mask.shape[0] * target_obj_mask.shape[1]) > 0.0001:
            assert len(detection_list) == len(gobs['mask']), f"Error in update_scene_graph: {len(detection_list)} != {len(gobs['mask'])}"  # sanity check

            # loop through the detected objects to find the highest IoU with the target object
            max_iou = -1
            max_iou_obj_id = None
            for idx, obj_id in enumerate(detection_list.keys()):
                detected_mask = gobs['mask'][idx]
                iou_score = IoU(detected_mask, target_obj_mask)
                if iou_score > max_iou:
                    max_iou = iou_score
                    max_iou_obj_id = obj_id
            print(f'!!!!!! {max_iou}')
            if max_iou > self.cfg.scene_graph.target_obj_iou_threshold:
                target_obj_id = max_iou_obj_id
                logging.info(f"Target object {target_obj_id} {detection_list[target_obj_id]['class_name']} detected with IoU {max_iou} in {img_path}!!!")


        # if there exists object detected in this frame, create a snapshot
        snapshot = SnapShot(
            image=img_path,
            color=(random.random(), random.random(), random.random()),
            obs_point=pts_voxel,
        )
        # add all detected objects into the snapshot
        snapshot.full_obj_list = {
            obj_id: detection_list[obj_id]['conf']
            for obj_id in detection_list.keys()
        }

        # if no objects yet in the map,
        # just add all the objects from the current frame
        # then continue, no need to match or merge
        if len(self.objects) == 0:
            logging.debug(f"No objects in the map yet, adding all detections of length {len(detection_list)}")
            self.objects.update(detection_list)

            # in this case, all objects are selected into this snapshot
            snapshot.selected_obj_list = list(detection_list.keys())

            self.snapshots[img_path] = snapshot

            return image_rgb, True, target_obj_id

        ### compute similarities and then merge
        spatial_sim = compute_spatial_similarities(
            spatial_sim_type=self.cfg_cg['spatial_sim_type'],
            detection_list=detection_list,
            objects=self.objects,
            downsample_voxel_size=self.cfg_cg['downsample_voxel_size']
        )

        visual_sim = compute_visual_similarities(detection_list, self.objects)

        agg_sim = aggregate_similarities(
            match_method=self.cfg_cg['match_method'],
            phys_bias=self.cfg_cg['phys_bias'],
            spatial_sim=spatial_sim,
            visual_sim=visual_sim
        )

        # Perform matching of detections to existing objects
        match_indices = match_detections_to_objects(
            agg_sim=agg_sim,
            detection_threshold=self.cfg_cg['sim_threshold'],  # Use the sim_threshold from the configuration
            existing_obj_ids=list(self.objects.keys()),
            detected_obj_ids=list(detection_list.keys())
        )

        # Now merge the detected objects into the existing objects based on the match indices
        visualize_captions, target_obj_id = self.merge_obj_matches(
            detection_list=detection_list,
            match_indices=match_indices,
            obj_classes=obj_classes,
            snapshot=snapshot,
            target_obj_id=target_obj_id
        )

        # add the snapshot into the snapshot list
        self.snapshots[img_path] = snapshot

        # create a Detection object for visualization
        det_visualize = sv.Detections(
            xyxy=gobs['xyxy'],
            confidence=gobs['confidence'],
            class_id=gobs['class_id'],
        )
        det_visualize.data['class_name'] = visualize_captions
        annotated_image = image_rgb.copy()
        BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
        LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK)
        annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, det_visualize)
        annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, det_visualize)

        return annotated_image, True, target_obj_id


    def filter_gobs_with_distance(self, pts, gobs):
        idx_to_keep = []
        for idx in range(len(gobs["bbox"])):
            if gobs["bbox"][idx] is None:  # point cloud was discarded
                continue

            # get the distance between the object and the current observation point
            if np.linalg.norm(
                np.asarray(gobs["bbox"][idx].center[0] - pts[0], gobs["bbox"][idx].center[2] - pts[2])
            ) > self.cfg.scene_graph.obj_include_dist:
                logging.debug(f"Object {gobs['detection_class_labels'][idx]} is too far away, skipping")
                continue
            idx_to_keep.append(idx)

        for attribute in gobs.keys():
            if isinstance(gobs[attribute], str) or attribute == "classes":  # Captions
                continue
            if attribute in ['labels', 'edges', 'text_feats', 'captions']:
                # Note: this statement was used to also exempt 'detection_class_labels' but that causes a bug. It causes the edges to be misalgined with the objects.
                continue
            elif isinstance(gobs[attribute], list):
                gobs[attribute] = [gobs[attribute][i] for i in idx_to_keep]
            elif isinstance(gobs[attribute], np.ndarray):
                gobs[attribute] = gobs[attribute][idx_to_keep]
            else:
                raise NotImplementedError(f"Unhandled type {type(gobs[attribute])}")

        return gobs

    def merge_obj_matches(
        self,
        detection_list: DetectionDict,
        match_indices: List[Tuple[int, Optional[int]]],
        obj_classes: ObjectClasses,
        snapshot: SnapShot,
        target_obj_id: Optional[int] = None  # if given, then track whether the target object is merged into a previous object (so the id would change)
    ) -> Tuple[List[str], Optional[int]]:
        visualize_captions = []
        for idx, (detected_obj_id, existing_obj_match_id) in enumerate(match_indices):
            if existing_obj_match_id is None:
                self.objects[detected_obj_id] = detection_list[detected_obj_id]
                visualize_captions.append(
                    f"{detected_obj_id} {self.objects[detected_obj_id]['class_name']} {self.objects[detected_obj_id]['conf']:.3f} N"
                )

                # add the object into the selected object list of the snapshot, since there's no previous observation of it
                snapshot.selected_obj_list.append(detected_obj_id)
            else:
                # merge detected object into existing object
                detected_obj = detection_list[detected_obj_id]
                matched_obj = self.objects[existing_obj_match_id]

                # check the confidence to see whether to select the object into the snapshot
                if detected_obj['conf'] > matched_obj['conf']:
                    old_snapshot_filename = matched_obj['image']
                    # replace the snapshot the old object belongs to
                    matched_obj['image'] = detected_obj['image']
                    # remove the object id in the old snapshot and add it to the new snapshot
                    self.snapshots[old_snapshot_filename].selected_obj_list.remove(existing_obj_match_id)
                    snapshot.selected_obj_list.append(existing_obj_match_id)

                merged_obj = merge_obj2_into_obj1(
                    obj1=matched_obj,
                    obj2=detected_obj,
                    downsample_voxel_size=self.cfg_cg['downsample_voxel_size'],
                    dbscan_remove_noise=self.cfg_cg['dbscan_remove_noise'],
                    dbscan_eps=self.cfg_cg['dbscan_eps'],
                    dbscan_min_points=self.cfg_cg['dbscan_min_points'],
                    spatial_sim_type=self.cfg_cg['spatial_sim_type'],
                    device=self.cfg_cg['device'],
                    run_dbscan=False,
                )
                # fix the class name by adopting the most popular class name
                class_id_counter = Counter(merged_obj['class_id'])
                most_common_class_id = class_id_counter.most_common(1)[0][0]
                most_common_class_name = obj_classes.get_classes_arr()[most_common_class_id]
                merged_obj['class_name'] = most_common_class_name

                # adjust the full detected list of the current snapshot: remove the detected object and add the merged object
                snapshot.full_obj_list[existing_obj_match_id] = detected_obj['conf']
                snapshot.full_obj_list.pop(detected_obj_id)

                self.objects[existing_obj_match_id] = merged_obj
                visualize_captions.append(
                    f"{existing_obj_match_id} {self.objects[existing_obj_match_id]['class_name']} {self.objects[existing_obj_match_id]['conf']:.3f} {merged_obj['num_detections']}"
                )

                # if current object is the target object, and it is merged into an existing object, then change the target object id to the existing object id
                if target_obj_id == detected_obj_id:
                    target_obj_id = existing_obj_match_id

        return visualize_captions, target_obj_id

    def make_detection_list_from_pcd_and_gobs(
            self, gobs, image_path, obj_classes
    ) -> DetectionDict:
        detection_list = DetectionDict()
        for mask_idx in range(len(gobs['mask'])):
            if gobs['pcd'][mask_idx] is None:  # point cloud was discarded
                continue

            curr_class_name = gobs['classes'][gobs['class_id'][mask_idx]]
            curr_class_idx = obj_classes.get_classes_arr().index(curr_class_name)

            detected_object = {
                'id': self.object_id_counter,  # unique id for this object

                'class_name': curr_class_name,  # global class id for this detection
                'class_id': [curr_class_idx],  # global class id for this detection
                'num_detections': 1,  # number of detections in this object
                'conf': gobs['confidence'][mask_idx],

                # These are for the entire 3D object
                'pcd': gobs['pcd'][mask_idx],
                'bbox': gobs['bbox'][mask_idx],
                'clip_ft': to_tensor(gobs['image_feats'][mask_idx]),

                # the snapshot name it belongs to
                'image': image_path,
            }

            detection_list[self.object_id_counter] = detected_object
            self.object_id_counter += 1

        return detection_list

    def cleanup_empty_snapshots(self):
        # remove the snapshots that have no selected objects
        filtered_snapshots = {}
        for file_name, snapshot in self.snapshots.items():
            if len(snapshot.selected_obj_list) > 0:
                filtered_snapshots[file_name] = snapshot
        self.snapshots = filtered_snapshots

        # for debug
        for file_name, snapshot in self.snapshots.items():
            for obj_id in snapshot.selected_obj_list:
                assert self.objects[obj_id]['image'] == file_name, f"Error in update_snapshots: {obj_id}: {self.objects[obj_id]['image']} != {file_name}"

    def update_snapshots(
        self,
        min_num_obj_threshold=1
    ):
        self.cleanup_empty_snapshots()

        for min_num_objs in range(1, min_num_obj_threshold + 1):
            for file_name, snapshot in self.snapshots.items():
                if len(snapshot.selected_obj_list) == min_num_objs:
                    # clean up this snapshot by moving its selected objects to other snapshots with have also observed them
                    # find the other snapshot that has the highest confidence for the object
                    remaining_ojb_ids = []
                    for obj_id in snapshot.selected_obj_list:
                        max_confidence = 0
                        max_confidence_file_name = None
                        for other_file_name, other_snapshot in self.snapshots.items():
                            if other_file_name == file_name:
                                continue
                            if obj_id in other_snapshot.full_obj_list.keys() and other_snapshot.full_obj_list[obj_id] > max_confidence:
                                max_confidence = other_snapshot.full_obj_list[obj_id]
                                max_confidence_file_name = other_file_name

                        # if there exists a snapshot that the object can be added to
                        if max_confidence_file_name is not None:
                            # move the object to that snapshot
                            self.objects[obj_id]['image'] = max_confidence_file_name
                            self.objects[obj_id]['conf'] = max_confidence
                            self.snapshots[max_confidence_file_name].selected_obj_list.append(obj_id)
                        else:
                            # otherwise, keep the object in this snapshot
                            remaining_ojb_ids.append(obj_id)
                    snapshot.selected_obj_list = remaining_ojb_ids
            self.cleanup_empty_snapshots()

    def periodic_cleanup_objects(self, frame_idx, pts):
        ### Perform post-processing periodically if told so

        # Denoising
        if processing_needed(
                self.cfg_cg["denoise_interval"],
                self.cfg_cg["run_denoise_final_frame"],
                frame_idx,
                is_final_frame=False,
        ):
            self.objects = measure_time(denoise_objects)(
                downsample_voxel_size=self.cfg_cg['downsample_voxel_size'],
                dbscan_remove_noise=self.cfg_cg['dbscan_remove_noise'],
                dbscan_eps=self.cfg_cg['dbscan_eps'],
                dbscan_min_points=self.cfg_cg['dbscan_min_points'],
                spatial_sim_type=self.cfg_cg['spatial_sim_type'],
                device=self.cfg_cg['device'],
                objects=self.objects
            )

        # Filtering
        if processing_needed(
                self.cfg_cg["filter_interval"],
                self.cfg_cg["run_filter_final_frame"],
                frame_idx,
                is_final_frame=False,
        ):
            self.objects = filter_objects(
                obj_min_points=self.cfg_cg['obj_min_points'],
                obj_min_detections=self.cfg_cg['obj_min_detections'],
                min_distance=self.cfg.scene_graph.obj_include_dist,
                objects=self.objects,
                pts=pts,
            )

        # temporarily we do not merge close objects, since handling which snapshot the merged object belongs to is a bit tricky

        # Merging
        if processing_needed(
                self.cfg_cg["merge_interval"],
                self.cfg_cg["run_merge_final_frame"],
                frame_idx,
                is_final_frame=False,
        ):
            self.objects = measure_time(merge_objects)(
                merge_overlap_thresh=self.cfg_cg["merge_overlap_thresh"],
                merge_visual_sim_thresh=self.cfg_cg["merge_visual_sim_thresh"],
                merge_text_sim_thresh=self.cfg_cg["merge_text_sim_thresh"],
                objects=self.objects,
                downsample_voxel_size=self.cfg_cg["downsample_voxel_size"],
                dbscan_remove_noise=self.cfg_cg["dbscan_remove_noise"],
                dbscan_eps=self.cfg_cg["dbscan_eps"],
                dbscan_min_points=self.cfg_cg["dbscan_min_points"],
                spatial_sim_type=self.cfg_cg["spatial_sim_type"],
                device=self.cfg_cg["device"],
            )

        # update the object list in snapshots, since some objects may have been removed
        for ss in self.snapshots.values():
            ss.selected_obj_list = [obj_id for obj_id in ss.selected_obj_list if obj_id in self.objects.keys()]
            ss.full_obj_list = {obj_id: self.objects[obj_id]['conf'] for obj_id in ss.full_obj_list.keys() if obj_id in self.objects.keys()}

















