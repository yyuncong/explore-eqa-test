import numpy as np
import random
import scipy.ndimage as ndimage
import heapq
import math
import logging
from sklearn.cluster import DBSCAN


def pos_habitat_to_normal(pts):
    # -90 deg around x-axis
    return np.dot(pts, np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))


def get_scene_bnds(pathfinder, floor_height):
    # Get mesh boundaries - this is for the full scene
    scene_bnds = pathfinder.get_bounds()
    scene_lower_bnds_normal = pos_habitat_to_normal(scene_bnds[0])
    scene_upper_bnds_normal = pos_habitat_to_normal(scene_bnds[1])
    scene_size = np.abs(
        np.prod(scene_upper_bnds_normal[:2] - scene_lower_bnds_normal[:2])
    )
    tsdf_bnds = np.array(
        [
            [
                min(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
                max(scene_lower_bnds_normal[0], scene_upper_bnds_normal[0]),
            ],
            [
                min(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
                max(scene_lower_bnds_normal[1], scene_upper_bnds_normal[1]),
            ],
            [
                floor_height - 0.2,
                floor_height + 3.5,
            ],
        ]
    )
    return tsdf_bnds, scene_size


def get_cam_intr(hfov, img_height, img_width):
    hfov_rad = hfov * np.pi / 180
    vfov_rad = 2 * np.arctan(np.tan(hfov_rad / 2) * img_height / img_width)
    # vfov = vfov_rad * 180 / np.pi
    fx = (1.0 / np.tan(hfov_rad / 2.0)) * img_width / 2.0
    fy = (1.0 / np.tan(vfov_rad / 2.0)) * img_height / 2.0
    cx = img_width // 2
    cy = img_height // 2
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype="int")  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float("inf")  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
            (points[last_added] - points[points_left]) ** 2
        ).sum(
            -1
        )  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(
            dist_to_last_added_point, dists[points_left]
        )  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds]


def points_in_circle(center_x, center_y, radius, grid_shape):
    x, y = np.meshgrid(
        np.arange(grid_shape[0]), np.arange(grid_shape[1]), indexing="ij"
    )  # use matrix indexing instead of cartesian
    distance_matrix = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    points_within_circle = np.where(distance_matrix <= radius)
    return list(zip(points_within_circle[0], points_within_circle[1]))


def run_dijkstra(grid, start, end):
    start = tuple(start[:2])
    end = tuple(end[:2])
    rows, cols = grid.shape
    directions = [
        (0, 1),
        (1, 0),
        (0, -1),
        (-1, 0),
        (1, 1),
        (1, -1),
        (-1, 1),
        (-1, -1),
    ]  # 8 directions
    distance = np.full(grid.shape, np.inf)
    distance[start] = 0
    prev = {start: None}

    # Priority queue for the nodes to explore
    pq = [(0, start)]  # (distance, node)

    while pq:
        dist, current = heapq.heappop(pq)

        if current == end:
            break

        for direction in directions:
            r, c = current[0] + direction[0], current[1] + direction[1]
            if 0 <= r < rows and 0 <= c < cols and grid[r][c] == 0:
                new_dist = dist + math.sqrt(direction[0] ** 2 + direction[1] ** 2)
                if new_dist < distance[r, c]:
                    distance[r, c] = new_dist
                    heapq.heappush(pq, (new_dist, (r, c)))
                    prev[(r, c)] = current

    # Reconstructing path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = prev.get(current)
    return path[::-1]  # Return reversed path


def find_normal(grid, x, y):
    # Sobel operators
    sobel_y = ndimage.sobel(grid, axis=0)  # gradient on y
    sobel_x = ndimage.sobel(grid, axis=1)  # gradient on x

    # Gradient at (x,y)
    Gx = sobel_x[x, y]
    Gy = sobel_y[x, y]

    # # If the gradient is near zero, we might be in a flat region or not on an edge
    if Gx == 0 and Gy == 0:
        # random
        normal = np.array([random.random(), random.random()])
    else:
        # Normal of the edge is perpendicular to the gradient
        normal = np.array([-Gy, -Gx])  # need to negate x???

    # Normalize the vector (make its length 1)
    normal = normal / np.linalg.norm(normal)
    return normal


def open_operation(array, structure=None):
    eroded = ndimage.binary_erosion(array, structure=structure)
    opened = ndimage.binary_dilation(eroded, structure=structure)
    return opened


def close_operation(array, structure=None):
    dilated = ndimage.binary_dilation(array, structure=structure)
    closed = ndimage.binary_erosion(dilated, structure=structure)
    return closed


def rigid_transform(xyz, transform):
    """Applies a rigid transform to an (N, 3) pointcloud."""
    xyz_h = np.hstack([xyz, np.ones((len(xyz), 1), dtype=np.float32)])
    xyz_t_h = np.dot(transform, xyz_h.T).T
    return xyz_t_h[:, :3]


def get_view_frustum(depth_im, cam_intr, cam_pose):
    """Get corners of 3D camera view frustum of depth image"""
    im_h = depth_im.shape[0]
    im_w = depth_im.shape[1]
    max_depth = np.max(depth_im)
    view_frust_pts = np.array(
        [
            (np.array([0, 0, 0, im_w, im_w]) - cam_intr[0, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[0, 0],
            (np.array([0, 0, im_h, 0, im_h]) - cam_intr[1, 2])
            * np.array([0, max_depth, max_depth, max_depth, max_depth])
            / cam_intr[1, 1],
            np.array([0, max_depth, max_depth, max_depth, max_depth]),
        ]
    )
    view_frust_pts = rigid_transform(view_frust_pts.T, cam_pose).T
    return view_frust_pts


def meshwrite(filename, verts, faces, norms, colors):
    """Save a 3D mesh to a polygon .ply file."""
    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (verts.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property float nx\n")
    ply_file.write("property float ny\n")
    ply_file.write("property float nz\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("element face %d\n" % (faces.shape[0]))
    ply_file.write("property list uchar int vertex_index\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(verts.shape[0]):
        ply_file.write(
            "%f %f %f %f %f %f %d %d %d\n"
            % (
                verts[i, 0],
                verts[i, 1],
                verts[i, 2],
                norms[i, 0],
                norms[i, 1],
                norms[i, 2],
                colors[i, 0],
                colors[i, 1],
                colors[i, 2],
            )
        )

    # Write face list
    for i in range(faces.shape[0]):
        ply_file.write("3 %d %d %d\n" % (faces[i, 0], faces[i, 1], faces[i, 2]))

    ply_file.close()


def pcwrite(filename, xyzrgb):
    """Save a point cloud to a polygon .ply file."""
    xyz = xyzrgb[:, :3]
    rgb = xyzrgb[:, 3:].astype(np.uint8)

    # Write header
    ply_file = open(filename, "w")
    ply_file.write("ply\n")
    ply_file.write("format ascii 1.0\n")
    ply_file.write("element vertex %d\n" % (xyz.shape[0]))
    ply_file.write("property float x\n")
    ply_file.write("property float y\n")
    ply_file.write("property float z\n")
    ply_file.write("property uchar red\n")
    ply_file.write("property uchar green\n")
    ply_file.write("property uchar blue\n")
    ply_file.write("end_header\n")

    # Write vertex list
    for i in range(xyz.shape[0]):
        ply_file.write(
            "%f %f %f %d %d %d\n"
            % (
                xyz[i, 0],
                xyz[i, 1],
                xyz[i, 2],
                rgb[i, 0],
                rgb[i, 1],
                rgb[i, 2],
            )
        )


def get_nearest_true_point(point, bool_map):
    """
    point: [2] array
    bool_map: [H, W] boolean array
    """
    H, W = bool_map.shape
    x, y = point
    x, y = int(x), int(y)
    if x < 0 or x >= H or y < 0 or y >= W:
        logging.error(f"Error in get_nearest_true_point: point {point} is out of the map of shape {bool_map.shape}")
        return None
    if bool_map[x, y]:
        return point
    horizontal_found = False
    for i in range(1, max(H, W)):
        for j in range(i + 1):
            for dx, dy in [(j, i), (j, -i), (-j, i), (-j, -i)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and bool_map[x + dx, y + dy]:
                    horizontal_found = True
                    break
            if horizontal_found:
                break
        if horizontal_found:
            break

    dx_horizon, dy_horizon = dx, dy
    vertical_found = False
    for i in range(1, max(H, W)):
        for j in range(i + 1):
            for dx, dy in [(i, j), (-i, j), (i, -j), (-i, -j)]:
                if 0 <= x + dx < H and 0 <= y + dy < W and bool_map[x + dx, y + dy]:
                    vertical_found = True
                    break
            if vertical_found:
                break
        if vertical_found:
            break
    dx_vertical, dy_vertical = dx, dy

    if not horizontal_found and not vertical_found:
        logging.error(f"Error in get_nearest_true_point: no true point found in the map of shape {bool_map.shape}")
        return None
    elif not horizontal_found:
        return np.array([x + dx_vertical, y + dy_vertical])
    elif not vertical_found:
        return np.array([x + dx_horizon, y + dy_horizon])
    else:
        if dx_horizon ** 2 + dy_horizon ** 2 < dx_vertical ** 2 + dy_vertical ** 2:
            return np.array([x + dx_horizon, y + dy_horizon])
        else:
            return np.array([x + dx_vertical, y + dy_vertical])


def get_proper_observe_point(point, unoccupied_map, cur_point, dist=10):
    # Get a proper observation point at dist
    # the observation point should not near the wall, so it's a bit tricky
    # point, dist are all in voxel space
    unoccupied_coords = np.argwhere(unoccupied_map)  # [N, 2]
    dists = np.linalg.norm(unoccupied_coords - point, axis=1)  # [N]
    valid_coords = unoccupied_coords[dists < dist]  # [N, 2]

    # cluster the points
    if len(valid_coords) == 0:
        logging.error(f"Error in get_proper_observe_point: no unoccupied points for {dist} distance around point {point}")
        return None
    clustering = DBSCAN(eps=1, min_samples=1).fit(valid_coords)
    labels = clustering.labels_
    unique_labels = np.unique(labels)
    # get the largest cluster
    max_cluster_id = -1
    max_cluster_size = 0
    for cluster_id in unique_labels:
        cluster_size = np.sum(labels == cluster_id)
        if cluster_size > max_cluster_size:
            max_cluster_size = cluster_size
            max_cluster_id = cluster_id
    if max_cluster_id == -1:
        logging.error(f"Error in get_proper_observe_point: clustering failed for {dist} distance around point {point}")
        return None
    max_cluster_coords = valid_coords[labels == max_cluster_id]
    max_cluster_center = np.mean(max_cluster_coords, axis=0)

    direction = max_cluster_center - point

    if np.linalg.norm(direction) < 1e-3:
        # if the surrounding of the object point is totally navigable
        direction = cur_point[:2] - point
        if np.linalg.norm(direction) < 1e-3:
            logging.error(f"Error in get_proper_observe_point: {max_cluster_center}, {point}, {cur_point}")
            return None

    direction = direction / np.linalg.norm(direction)
    final_point = point + direction * dist

    H, W = unoccupied_map.shape
    final_point = np.clip(final_point, [0, 0], [H - 1, W - 1])

    # ensure the final point is navigable
    final_point = get_nearest_true_point(final_point, unoccupied_map)
    return final_point


def get_proper_observe_point_with_pathfinder(target_point_habitat, pathfinder, height):
    # target_point_habitat: [3] array in habitat coordinate
    # return the proper observation point in habitat coordinate
    try_count = 0
    while True:
        try_count += 1
        if try_count > 100:
            logging.error(f"Error in get_proper_observe_point_with_pathfinder: cannot find a proper observation point! try many tries")
            return None

        try:
            target_navigable_point_habitat = pathfinder.get_random_navigable_point_near(
                circle_center=target_point_habitat,
                radius=1.0,
            )
        except:
            logging.error(f"Error in get_proper_observe_point_with_pathfinder: pathfinder failed to find a navigable point!")
            continue
        if np.isnan(target_navigable_point_habitat).any():
            logging.error(f"Error in get_proper_observe_point_with_pathfinder: pathfinder returned nan point!")
            continue
        if abs(target_navigable_point_habitat[1] - height) < 0.1:
            return target_navigable_point_habitat

    logging.error(f"Error in get_proper_observe_point_with_pathfinder: cannot find a proper observation point!")
    return None



def get_random_observe_point(point, unoccupied_map, min_dist=15, max_dist=30):
    # Get a random observation point between min_dist and max_dist around point
    # there shouldn't be obstacles between the point and the observation point
    unoccupied_coords = np.argwhere(unoccupied_map)  # [N, 2]
    dists = np.linalg.norm(unoccupied_coords - point, axis=1)  # [N]
    valid_coords = unoccupied_coords[(dists > min_dist) & (dists < max_dist)]  # [N, 2]

    if len(valid_coords) == 0:
        logging.error(f"Error in get_random_observe_point: no unoccupied points for {min_dist}-{max_dist} distance around point {point}")
        return None
    try_count = 0
    while True:
        try_count += 1
        # randomly pick a point, and check its validity
        idx = random.randint(0, len(valid_coords) - 1)
        potential_obs_point = valid_coords[idx]
        # check whether there are false points between the obs point and the target point
        direction = point - potential_obs_point
        direction = direction / np.linalg.norm(direction)
        # adjust the point: usually there are surrounding occupied points around the target object
        target_point_adjusted = point.copy()
        try_count_step_back = 0
        adjust_success = True
        while not unoccupied_map[int(target_point_adjusted[0]), int(target_point_adjusted[1])]:
            try_count_step_back += 1
            target_point_adjusted = target_point_adjusted - direction
            if try_count_step_back > max_dist * 2:
                logging.error(f"Error in get_random_observe_point: cannot backtrace from {point} to {potential_obs_point}!")
                adjust_success = False
                break
        if not adjust_success:
            continue
        target_point_adjusted = target_point_adjusted.astype(int)
        direction = target_point_adjusted - potential_obs_point

        if check_distance(
            occupied_map=np.logical_not(unoccupied_map),
            pos=potential_obs_point,
            direction=direction / np.linalg.norm(direction),
            tolerance=int(np.linalg.norm(direction)) - 1
        ):
            break

        if try_count > 1000:
            logging.error(f"Error in get_random_observe_point: cannot find a proper observation point! try many tries")
            return None

    return potential_obs_point





def get_warping_gap(angles, tolerance_degree=30, max_try=1000):
    # angles: [N], should be sorted
    tolerance = tolerance_degree / 180 * np.pi
    angles = np.asarray(angles)
    angles = np.sort(angles)
    if angles[-1] - angles[0] > 1.95 * np.pi:
        # need to wrap the group of angles
        # randomly select one angle that perfectly splits the angles into two groups
        ang = None
        for _ in range(max_try):
            ang = random.uniform(-np.pi, np.pi)
            if np.sum((ang - tolerance / 2 < angles) & (angles < ang + tolerance / 2)) == 0:
                break
        return ang
    else:
        # no need to wrap
        return None

def get_angle_span(angles):
    # angles: [N], range from -pi to pi
    normalized_angles = [(angle + 2 * np.pi) % (2 * np.pi) for angle in angles]
    normalized_angles = np.sort(normalized_angles)

    differences = np.diff(normalized_angles)
    differences = np.append(differences, normalized_angles[0] + 2 * np.pi - normalized_angles[-1])

    max_gap = np.max(differences)

    return 2 * np.pi - max_gap


def adjust_navigation_point(pos, occupied, max_dist=0.5, max_adjust_distance=0.3, step_size=0.05, voxel_size=0.1):
    # adjust pos a bit to make it not too close to the occupied area, avoiding bad observations
    max_dist = max_dist / voxel_size
    max_adjust_distance = max_adjust_distance / voxel_size
    step_size = step_size / voxel_size
    pos = pos.astype(int)

    if occupied[pos[0], pos[1]] == 1:
        # if the current position is occupied, we need to first find a nearby unoccupied position
        pos = get_nearest_true_point(pos, np.logical_not(occupied))

    # find the nearest occupied point
    occupied_point = get_nearest_true_point(pos, occupied)
    direction = occupied_point - pos
    original_dist = np.linalg.norm(direction)
    direction = direction / original_dist

    if original_dist > max_dist:
        # if the point is far from the occupied point, we don't need to adjust it
        return pos

    # adjust the point
    min_dist = original_dist
    new_pos = pos
    max_try = 100
    count = 0
    while count < max_try:
        count += 1
        new_pos = new_pos - direction * step_size
        new_pos_int = np.round(new_pos).astype(int)
        new_occupied_point = get_nearest_true_point(new_pos_int, occupied)
        new_dist = np.linalg.norm(new_occupied_point - new_pos_int)
        if new_dist >= max_dist:
            break
        if np.linalg.norm(new_pos_int - pos) >= max_adjust_distance:
            break
        if new_dist >= min_dist:
            min_dist = new_dist
        else:
            new_pos = new_pos + direction * step_size
            new_pos_int = np.round(new_pos).astype(int)
            break
        # update direction
        direction = new_occupied_point - new_pos_int
        direction = direction / np.linalg.norm(direction)

    return new_pos_int


def check_distance(occupied_map, pos, direction, tolerance):
    # occupied_map, pos, direction, and tolerance are all in voxel space
    if tolerance <= 0:
        return False
    max_steps = tolerance
    all_points = np.round(
        np.linspace(pos, pos + direction * max_steps, max_steps + 1)
    ).astype(int)
    for point in all_points:
        if occupied_map[point[0], point[1]]:
            return False
    return True


def get_collision_distance(occupied_map, pos, direction, max_step=50):
    # occupied_map, pos, and direction are all in voxel space
    pos = pos[:2]
    curr_pos = pos
    count = 0
    scene_bound = occupied_map.shape
    while count < max_step:
        count += 1
        curr_pos = curr_pos + direction
        if 0 <= curr_pos[0] < scene_bound[0] and 0 <= curr_pos[1] < scene_bound[1] and occupied_map[int(curr_pos[0]), int(curr_pos[1])]:
            break
    curr_pos = curr_pos.astype(int)
    return np.linalg.norm(curr_pos - pos)


def region_equal(region_1, region_2, threshold=0.98):
    # region 1, 2: boolean array of the same shape
    # threshold: the number of different points
    # tolenrance_1 = max(min_threshold, np.sum(region_1) // 100)
    # diff_1 = np.sum(region_1) - np.sum(region_1 & region_2)
    # tolerance_2 = max(min_threshold, np.sum(region_2) // 100)
    # diff_2 = np.sum(region_2) - np.sum(region_1 & region_2)
    # return tolenrance_1 < diff_1 and tolerance_2 < diff_2
    return IoU(region_1, region_2) > threshold


def IoU(region_1, region_2):
    # region 1, 2: boolean array of the same shape
    intersection = np.sum(region_1 & region_2)
    union = np.sum(region_1 | region_2)
    return intersection / union


def get_proper_step_length(dist, min_length, max_length, min_threshold, max_threshold):
    if dist < min_threshold:
        return min_length
    elif dist > max_threshold:
        return max_length
    else:
        return min_length + (max_length - min_length) * (dist - min_threshold) / (max_threshold - min_threshold)


def pix_diff(region_1, region_2):
    # region 1, 2: boolean array of the same shape
    return np.sum(region_1 | region_2) - np.sum(region_1 & region_2)


def get_proper_snapshot_observation_point(
        obj_centers,
        snapshot_observation_point,
        unoccupied_map,
        min_obs_dist=10,
        max_obs_dist=15,
):
    # obj_centers: [N, 2] in voxel space
    # unoccupied_map: [H, W] boolean array
    # obs_dist: the distance between the observation point and the snapshot center in voxel space

    snapshot_center = np.mean(obj_centers, axis=0)
    if len(obj_centers) == 2:
        # if there are two objects in the snapshot, then the long axis is the line bewteen the two objects
        # and the short axis is the perpendicular line to the long axis
        long_axis = obj_centers[1] - obj_centers[0]
        long_axis = long_axis / np.linalg.norm(long_axis)
        short_axis = np.array([-long_axis[1], long_axis[0]])
    else:
        obj_centers = obj_centers - snapshot_center
        structure_tensor = np.cov(obj_centers.T)
        eigenvalues, eigenvecs = np.linalg.eig(structure_tensor)
        long_axis = eigenvecs[:, np.argmax(eigenvalues)]
        short_axis = eigenvecs[:, np.argmin(eigenvalues)]
        # when the object centers are generally on the same line, the short axis is very small
        if eigenvalues.min() < 1e-3:
            short_axis = np.array([-long_axis[1], long_axis[0]])
        short_axis = short_axis / np.linalg.norm(short_axis)


    # adjust the direction of the short axis
    ss_direction = snapshot_center - snapshot_observation_point[:2]
    if np.dot(ss_direction, short_axis) < 0:
        short_axis = -short_axis

    # get the points in unoccupied_map that are within the observation distance
    unoccupied_coords = np.argwhere(unoccupied_map)  # [N, 2]
    dists = np.linalg.norm(unoccupied_coords - snapshot_center, axis=1)  # [N]
    valid_coords = unoccupied_coords[(dists > min_obs_dist) & (dists < max_obs_dist)]  # [N, 2]
    if len(valid_coords) == 0:
        logging.error(f"Error in get_random_snapshot_observation_point: no unoccupied points for {min_obs_dist}-{max_obs_dist} distance around snapshot center {snapshot_center}")
        return None

    # get the point that is the most opposite to the short axis
    cos_values = np.dot(valid_coords - snapshot_center, short_axis) / np.linalg.norm(valid_coords - snapshot_center, axis=1)
    indices_rank  = np.argsort(cos_values)
    target_obs_point = None
    for idx in indices_rank:
        potential_obs_point = valid_coords[idx]

        # we need to ensure that there is no occupied point between the observation point and the snapshot center
        direction = snapshot_center - potential_obs_point
        direction = direction / np.linalg.norm(direction)
        # adjust the snapshot center: usually there are surrounding occupied regions around the snapshot center
        snapshot_center_adjusted = snapshot_center.copy()
        try_count_step_back = 0
        adjust_success = True
        while not unoccupied_map[int(snapshot_center_adjusted[0]), int(snapshot_center_adjusted[1])]:
            try_count_step_back += 1
            snapshot_center_adjusted = snapshot_center_adjusted - direction
            if try_count_step_back > max_obs_dist * 2:
                logging.error(f"Error in get_random_snapshot_observation_point: cannot backtrace from {snapshot_center} to {potential_obs_point}!")
                adjust_success = False
                break
        if not adjust_success:
            continue
        snapshot_center_adjusted = snapshot_center_adjusted.astype(int)
        direction = snapshot_center_adjusted - potential_obs_point

        # then we can check whether there are occupied points between the observation point and the adjusted snapshot center
        if check_distance(
            occupied_map=np.logical_not(unoccupied_map),
            pos=potential_obs_point,
            direction=direction / np.linalg.norm(direction),
            tolerance=int(np.linalg.norm(direction)) - 1
        ):
            target_obs_point = potential_obs_point
            break

    if target_obs_point is None:
        logging.error(f"Error in get_random_snapshot_observation_point: cannot find a proper observation point!")
        return None

    return target_obs_point





