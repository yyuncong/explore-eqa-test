from .geom import IoU
import numpy as np
def compute_recall(detected_scene_graph,gt_scene_graph):
    """
    Compute how many of the ground truth objects can be detected
    """
    detected_ids = set(detected_scene_graph.keys())
    gt_ids = set(gt_scene_graph.keys())
    if len(gt_ids) == 0:
        return 1.0
    return len(detected_ids.intersection(gt_ids)) / len(gt_ids)
    

def format_snapshot(snapshots, scene_graph):
    step_snapshots = []
    for snapshot in snapshots:
        snapshot_dict = {}
        snapshot_dict['img_id'] = snapshot.image
        snapshot_dict['obj_ids'] = {}
        for obj_id in snapshot.cluster:
            obj_id = int(obj_id)
            # extract the most common class name as the final class name
            class_name,count = '',0
            for class_name_,count_ in scene_graph[obj_id].classes.items():
                if count_ > count:
                    class_name = class_name_
                    count = count_
            snapshot_dict['obj_ids'][obj_id] = {
                'recognize_class': class_name,
                'gt_class': scene_graph[obj_id].gt_class,
            }
        step_snapshots.append(snapshot_dict)
    return step_snapshots

def detection_metric(iou,confidence,measure = "iou"):
    """
    Compute the detection metric based on the iou and confidence
    """
    if measure == "iou":
        return iou
    elif measure == "confidence":
        return confidence
    elif measure == "mixed":
        return iou*confidence
    else:
        raise ValueError("measure should be iou, confidence or iou_and_confidence")
    
def gt_class_match(
    detect_info,
    obj_ids,
    obj_id_to_bbox,
    semantic_obs,
    obs_point,
    obj_include_dist,
    iou_threshold):
    
    for obj_id in obj_ids:
        if obj_id not in obj_id_to_bbox.keys():
            continue
        obj_x_start, obj_y_start = np.argwhere(semantic_obs == obj_id).min(axis=0)
        obj_x_end, obj_y_end = np.argwhere(semantic_obs == obj_id).max(axis=0)
        obj_mask = np.zeros(semantic_obs.shape, dtype=bool)
        obj_mask[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = True
        if IoU(detect_info['bbox_mask'], obj_mask) > iou_threshold:
        # what is the caption for?
            caption = f"{obj_id}_{detect_info['class_name']}_{detect_info['confidence']:.3f}"

            # get the center of the bounding box to check whether it is close to the agent
            bbox = obj_id_to_bbox[obj_id]["bbox"]
            bbox = np.asarray(bbox)
            bbox_center = np.mean(bbox, axis=0)
            # change to x, z, y for habitat
            bbox_center = bbox_center[[0, 2, 1]]

            # if the object is faraway, then just not add to the scene graph
            if np.linalg.norm(np.asarray([bbox_center[0] - obs_point[0], bbox_center[2] - obs_point[2]])) > obj_include_dist:
                continue

            return obj_id, caption, bbox_center
            '''
            # this object is counted as detected
            frame.full_obj_list[obj_id] = confidence
            # add to the scene graph if it is not in the scene graph
            if obj_id not in self.simple_scene_graph.keys():
                # add to simple scene graph
                self.simple_scene_graph[obj_id] = SceneGraphItem(
                    object_id=obj_id,
                    bbox_center=bbox_center,
                    confidence=confidence,
                    image=None
                )
                if obj_id not in self.scene_graph_list:
                    self.scene_graph_list.append(obj_id)
                caption += "_N"


            adopted_indices.append(i)
            caption_list.append(caption)
            object_added = True
            detected_class.append(class_name)
            all_obj_id.remove(obj_id)
            break 
            '''
    return None, None, None