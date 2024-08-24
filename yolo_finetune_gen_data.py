import os
import random
import argparse

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
import matplotlib.pyplot as plt
import habitat_sim
import supervision as sv

from src.habitat import (
    make_semantic_cfg_new,
    get_quaternion,
)
from src.geom import get_cam_intr, IoU
from ultralytics import YOLO
from ultralytics.utils.ops import xyxy2xywh


def check_must_include(class_name: str, class_id, class_count: dict) -> bool:
    if class_name in ['refrigerator', 'fridge', 'oven', 'microwave', 'furnace', 'oven and stove']:
        return True
    if class_name in ['mirror', 'heater', 'tv', 'fireplace', 'table', 'sink', 'couch', 'clothes', 'hanging clothes', 'treadmill', 'dustbin']:
        return True
    if class_name in ['monitor', 'screen', 'printer']:
        return True
    if class_name.endswith(' table') or class_name.endswith(' lamp') or class_name.endswith(' bin'):
        return True
    if 'chair' in class_name or 'desk' in class_name or 'machine' in class_name or 'exercise' in class_name:
        return True
    if 'hood' in class_name:
        return True


    if class_count[str(class_id)]['count'] <= 6:
        print(f'Add rare class {class_name}!!!!')
        return True


    return False



def select_detections(detections: sv.Detections, class_id_to_class_name: dict):
    detection_to_exclude_ids = []
    for idx in range(len(detections)):
        class_name = class_id_to_class_name[detections.class_id[idx]]
        caption = detections.data["class_name"][idx]
        pred_class_name = caption.split(";")[0]

        # for some classes, we only keep the detections if the prediction is correct
        if class_name in ['door', 'curtain']:
            if pred_class_name != class_name:
                detection_to_exclude_ids.append(idx)
                print(f"Exclude {class_name} {pred_class_name}")

    detection_keep_ids = [i for i in range(len(detections)) if i not in detection_to_exclude_ids]
    detections = detections[detection_keep_ids]
    return detections



def main(args):
    class_id_to_class_name = json.load(open("yolo_finetune/class_id_to_class_name.json", "r"))
    class_id_to_class_name = {int(k): v for k, v in class_id_to_class_name.items()}
    class_name_to_class_id = {cls: i for i, cls in class_id_to_class_name.items()}
    all_classes = list(class_id_to_class_name.values())

    class_count = json.load(open("yolo_finetune/class_count.json", "r"))

    detection_model = YOLO('yolov8x-world.pt')

    img_height = 1280
    img_width = 1280
    hfov = 120
    camera_height = 1.5
    camera_tilt_deg = -30

    confidence_threshold = 0.05
    iou_threshold = 0.8
    hm3d_label_min_pix_ratio = 0.005
    min_mask_over_bbox = 0.3
    min_center_to_edge_ratio = 0.05

    seed = args.seed

    scene_path = '/gpfs/u/home/LMCG/LMCGnngn/scratch/multisensory/MLLM/data/hm3d'
    scene_dataset_config_path = "data/hm3d_annotated_basis.scene_dataset_config.json"

    dataset_save_dir = "yolo_finetune_data"
    num_obs_per_scene = args.n_obs





    image_save_dir = os.path.join(dataset_save_dir, "images")
    label_save_dir = os.path.join(dataset_save_dir, "labels")
    os.makedirs(dataset_save_dir, exist_ok=True)
    os.makedirs(image_save_dir, exist_ok=True)
    os.makedirs(label_save_dir, exist_ok=True)

    total_obs_count = len(os.listdir(image_save_dir))
    print(f"Total existing observations: {total_obs_count}")

    all_scene_ids = os.listdir(os.path.join(scene_path, 'train')) + os.listdir(os.path.join(scene_path, 'val'))
    all_scene_ids = [scene_id for scene_id in all_scene_ids if '-' in scene_id and scene_id.split("-")[0].isdigit()]
    # sort the scene ids
    all_scene_ids = sorted(all_scene_ids, key=lambda x: int(x.split("-")[0]))

    for scene_id in all_scene_ids:
        split = 'train' if int(scene_id.split("-")[0]) < 800 else 'val'

        scene_mesh_path = os.path.join(scene_path, split, scene_id, scene_id.split("-")[1] + ".basis.glb")
        navmesh_path = os.path.join(scene_path, split, scene_id, scene_id.split("-")[1] + ".basis.navmesh")
        semantic_texture_path = os.path.join(scene_path, split, scene_id, scene_id.split("-")[1] + ".semantic.glb")
        scene_semantic_annotation_path = os.path.join(scene_path, split, scene_id, scene_id.split("-")[1] + ".semantic.txt")
        if not os.path.exists(scene_mesh_path) or not os.path.exists(navmesh_path) or not os.path.exists(semantic_texture_path) or not os.path.exists(scene_semantic_annotation_path):
            print(f"{scene_mesh_path} or\n{navmesh_path} or\n{semantic_texture_path} or\n{scene_semantic_annotation_path} does not exist")
            continue

        try:
            simulator.close()
        except:
            pass

        sim_settings = {
            "scene": scene_mesh_path,
            "default_agent": 0,
            "sensor_height": camera_height,
            "width": img_width,
            "height": img_height,
            "hfov": hfov,
            "scene_dataset_config_file": scene_dataset_config_path,
            "camera_tilt": camera_tilt_deg * np.pi / 180,
        }
        sim_cfg = make_semantic_cfg_new(sim_settings)
        simulator = habitat_sim.Simulator(sim_cfg)
        pathfinder = simulator.pathfinder
        pathfinder.seed(seed)
        pathfinder.load_nav_mesh(navmesh_path)
        agent = simulator.initialize_agent(sim_settings["default_agent"])
        agent_state = habitat_sim.AgentState()
        print(f"Load scene {scene_id} successfully")

        # read scene classes
        obj_id_to_class_name = {}
        with open(scene_semantic_annotation_path, 'r') as f:
            all_lines = f.readlines()[1:]
            for line in all_lines:
                class_name = line.split(",")[2].replace("\"", "")
                obj_id = int(line.split(",")[0])
                obj_id_to_class_name[obj_id] = class_name

        # get how many observations have been generated for this scene
        scene_obs_count = len([f for f in os.listdir(image_save_dir) if f.split("--")[1].split(".")[0] == scene_id])

        while scene_obs_count < num_obs_per_scene:
            scene_obs_count += 1

            # get a random observation
            # simulator.pathfinder.seed(seed)
            nav_point = pathfinder.get_random_navigable_point()
            rand_angle = np.random.uniform(0, 2 * np.pi)

            agent_state.position = nav_point
            agent_state.rotation = get_quaternion(rand_angle, 0)
            agent.set_state(agent_state)
            obs = simulator.get_sensor_observations()

            rgb = obs["color_sensor"][..., :3]
            semantic = obs["semantic_sensor"]

            all_obj_id = np.unique(semantic)
            all_obj_id = [obj_id for obj_id in all_obj_id if obj_id in obj_id_to_class_name]
            all_obs_classes = [obj_id_to_class_name[obj_id] for obj_id in all_obj_id if obj_id != 0]
            all_obs_classes = list(set(all_obs_classes))
            all_obs_classes = [cls for cls in all_obs_classes if cls in all_classes]

            if len(all_obs_classes) == 0:
                scene_obs_count -= 1
                continue

            # detection
            detection_model.set_classes(all_obs_classes)
            results = detection_model.predict(rgb, conf=confidence_threshold, verbose=False)
            confidences = results[0].boxes.conf.cpu().numpy()
            detected_class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            detection_classes = [all_obs_classes[cls_id] for cls_id in detected_class_ids]
            class_ids = np.asarray([class_name_to_class_id[cls] for cls in detection_classes])
            xyxy_tensor = results[0].boxes.xyxy
            xyxy_np = xyxy_tensor.cpu().numpy()

            detections = sv.Detections(
                xyxy=xyxy_np,
                confidence=confidences,
                class_id=class_ids,
            )

            detection_infos = []
            for i in range(len(detections)):
                info = {}
                info['pred_class_name'] = detection_classes[i]
                info['confidence'] = confidences[i]
                x_start, y_start, x_end, y_end = detections.xyxy[i].astype(int)
                # set bbox mask
                bbox_mask = np.zeros_like(semantic, dtype=bool)
                bbox_mask[y_start:y_end, x_start:x_end] = True
                info['bbox_mask'] = bbox_mask
                detection_infos.append(info)

            adopted_indices = []
            caption_list = []
            new_detection_list = []
            for obj_id in all_obj_id:
                if obj_id == 0:
                    continue
                if obj_id_to_class_name[obj_id] not in all_classes:
                    continue

                # get the bbox mask from ground truth semantic map
                obj_x_start, obj_y_start = np.argwhere(semantic == obj_id).min(axis=0)
                obj_x_end, obj_y_end = np.argwhere(semantic == obj_id).max(axis=0)
                obj_mask = np.zeros_like(semantic, dtype=bool)
                obj_mask[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = True

                # find the most matched detection
                potential_matches = []
                for i, dinfo in enumerate(detection_infos):
                    if i in adopted_indices:
                        continue

                    iou = IoU(dinfo['bbox_mask'], obj_mask)

                    if dinfo['confidence'] > confidence_threshold and iou > iou_threshold:
                        potential_matches.append((iou, i))

                if len(potential_matches) == 0:
                    pix_ratio = np.sum(semantic == obj_id) / (img_height * img_width)
                    if pix_ratio < hm3d_label_min_pix_ratio:
                        print(f"{obj_id: 5d}: {pix_ratio: .5f}, {obj_id_to_class_name[obj_id]}, too small")
                        continue
                    if not check_must_include(obj_id_to_class_name[obj_id], class_name_to_class_id[obj_id_to_class_name[obj_id]], class_count):
                        print(f"{obj_id: 5d}: {pix_ratio: .5f}, {obj_id_to_class_name[obj_id]}, no match")
                        continue

                    # check the mask size with the bbox size
                    mask_over_bbox = np.sum(semantic == obj_id) / np.sum(obj_mask)
                    if mask_over_bbox < min_mask_over_bbox:
                        print(f"{obj_id: 5d}: {pix_ratio: .5f}, {obj_id_to_class_name[obj_id]}, mask too inconsistent! {mask_over_bbox: .5f}")
                        continue

                    # the forced added objects should not be in the periphery, since there would be significant perspective distortion
                    center = np.argwhere(semantic == obj_id).mean(axis=0) / np.array([img_height, img_width])
                    if obj_x_start == 0 or obj_x_end == img_height - 1 or obj_y_start == 0 or obj_y_end == img_width - 1:
                        x_center, y_center = center
                        if not (min_center_to_edge_ratio < x_center < 1 - min_center_to_edge_ratio and min_center_to_edge_ratio < y_center < 1 - min_center_to_edge_ratio):
                            print(f"{obj_id: 5d}: {pix_ratio: .5f}, {obj_id_to_class_name[obj_id]}, too close to the edge! {center}")
                            continue





                    # create a new detection directly based on the ground truth semantic map
                    new_detection = sv.Detections(
                        xyxy=np.array([obj_y_start, obj_x_start, obj_y_end, obj_x_end]).reshape(1, 4),
                        confidence=np.array([1.0]),
                        class_id=np.array([class_name_to_class_id[obj_id_to_class_name[obj_id]]]).astype(int),
                    )
                    new_detection.data["class_name"] = [f"{obj_id_to_class_name[obj_id]}: gt {center[0]:.3f}, {center[1]:.3f}"]
                    new_detection_list.append(new_detection)
                    print(f"{obj_id: 5d}: {pix_ratio: .5f}, {obj_id_to_class_name[obj_id]}, add new detection, {mask_over_bbox: .5f}, {center}")
                else:
                    print(f"{obj_id: 5d}: {np.sum(semantic == obj_id) / (img_height * img_width): .5f}, {obj_id_to_class_name[obj_id]}")

                    # sort by iou
                    potential_matches.sort(key=lambda x: x[0], reverse=True)
                    best_match = potential_matches[0]
                    adopted_indices.append(best_match[1])

                    matched_index = best_match[1]
                    # replace the detection bbox with the ground truth bbox
                    detections.xyxy[matched_index] = np.array([obj_y_start, obj_x_start, obj_y_end, obj_x_end])
                    # replace the detection class id with the ground truth class id
                    detections.class_id[matched_index] = class_name_to_class_id[obj_id_to_class_name[obj_id]]

                    matched_item = detection_infos[matched_index]
                    caption = f"{matched_item['pred_class_name']};{class_id_to_class_name[int(detections.class_id[matched_index])]}: {matched_item['confidence']:.3f}"
                    caption_list.append(caption)

            detections = detections[adopted_indices]
            detections.data["class_name"] = caption_list

            # further filter the detections
            detections = select_detections(detections, class_id_to_class_name)

            if len(new_detection_list) > 0:
                new_detection_list = sv.Detections.merge(new_detection_list)
                detections = sv.Detections.merge([detections, new_detection_list])
            # make sure the class id is integer
            detections.class_id = detections.class_id.astype(int)

            # change the caption for better visualization
            for i in range(len(detections)):
                class_id = int(detections.class_id[i])
                detections.data["class_name"][i] = f"{class_id}, {class_id_to_class_name[class_id]}"

            # annotate the image
            annotated_image = rgb.copy()
            BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator(thickness=1)
            LABEL_ANNOTATOR = sv.LabelAnnotator(text_thickness=1, text_scale=0.25, text_color=sv.Color.BLACK)
            annotated_image = BOUNDING_BOX_ANNOTATOR.annotate(annotated_image, detections)
            annotated_image = LABEL_ANNOTATOR.annotate(annotated_image, detections)

            # save the image and label
            img_save_path = os.path.join(image_save_dir, f"{total_obs_count:06d}--{scene_id}.png")
            plt.imsave(img_save_path, rgb)

            label_save_path = os.path.join(label_save_dir, f"{total_obs_count:06d}--{scene_id}.txt")
            with open(label_save_path, 'w') as f:
                for i in range(len(detections)):
                    class_id = detections.class_id[i]
                    xywh = xyxy2xywh(detections.xyxy[i])
                    xywh /= np.array([img_width, img_height, img_width, img_height])
                    f.write(f"{class_id} {xywh[0]:.6f} {xywh[1]:.6f} {xywh[2]:.6f} {xywh[3]:.6f}\n")

            total_obs_count += 1
            print(f"Scene {scene_id}, obs {scene_obs_count}/{num_obs_per_scene}, total obs {total_obs_count}")









if __name__ == "__main__":
    # add command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--n_obs", type=int, required=True)

    args = parser.parse_args()

    main(args)


















