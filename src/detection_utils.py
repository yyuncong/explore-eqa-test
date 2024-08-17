
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