from PIL import Image

import numpy as np
import torch

from kornia.geometry.linalg import compose_transformations, inverse_transformation

def transform_points_batch(poses: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """
    poses: (M, 4, 4)
    points: (N, 3)
    
    return: (M, N, 3)
    """
    N = points.shape[0]
    M = poses.shape[0]
    
    # Convert points to homogeneous coordinates (N, 4)
    points_homogeneous = torch.cat((points, torch.ones(N, 1).to(points.device)), dim=-1) # (N, 4)
    
    # Repeat points M times along a new dimension
    points_homogeneous = points_homogeneous.unsqueeze(0).repeat(M, 1, 1) # (M, N, 4)
    
    # Apply transformation: for each pose, do a matrix multiplication with points
    points_transformed = torch.bmm(poses, points_homogeneous.transpose(1,2)) # (M, 4, N)
    
    # Convert back to cartesian coordinates by dividing by the last coordinate
    points_transformed = points_transformed[:, :3, :] / points_transformed[:, 3:4, :] # (M, 3, N)
    
    return points_transformed.transpose(1,2) # (M, N, 3)

def project_points_camera2plane(points: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Project a set of points (in camera coordinates) to the image plane according to the camera intrinsics
    
    points: (N, 3)
    K: (3, 3)
    
    return: 
        points_proj: (N, 2) the projected points in the image plane
        points_depth: (N,) the depth of the points in the camera coordinates
    """
    # multiply points by the camera intrinsics
    points_proj = torch.matmul(K, points.t())  # result size is (3, N)
    
    # convert to Euclidean coordinates
    points_depth = points_proj[2, :]  # (N)
    points_coord = points_proj[:2, :] / points_proj[2, :].unsqueeze(0)  # divide x and y by z
    
    return points_coord.t(), points_depth # transpose back to (N, 2)

def project_points_world2plane(points: torch.Tensor, poses: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """Project a set of points (in the world coordinates) to the image plane according to the camera pose and intrinsics

    Args:
        points (torch.Tensor): (N, 3)
        poses (torch.Tensor): (M, 4, 4) the camera poses in the world coordinates
        K (torch.Tensor): (3, 3)

    Returns:
        points_coord (torch.Tensor): (M, N, 2) the projected points in the image plane
        points_depth (torch.Tensor): (M, N) the depth of the points in the camera coordinates
    """
    poses_inv = torch.inverse(poses)  # (M, 4, 4)
    
    # Transform points from world coordinates to camera coordinates
    points_camera = transform_points_batch(poses_inv, points)  # (M, N, 3)

    M, N, _ = points_camera.shape

    points_camera = points_camera.reshape(-1, 3)  # (M*N, 3)
    points_coord, points_depth = project_points_camera2plane(points_camera, K)  # (M*N, 2), (M, N)
    
    points_coord = points_coord.reshape(M, N, 2)  # (M, N, 2)
    points_depth = points_depth.reshape(M, N)  # (M, N)

    return points_coord, points_depth  # (M, N, 2), (M, N)
    
def check_proj_points(
    points: torch.Tensor, 
    depth_tensor: torch.Tensor,
    K: torch.Tensor, 
    pose: torch.Tensor,
    depth_margin: float = 0.05,
) -> torch.Tensor:
    '''
    Project points to the image plane and perform visibility checks
    
    Args:
        points (torch.Tensor): (N, 3), points in the world coordinates
        depth_tensor (torch.Tensor): (H, W)
        K (torch.Tensor): (3, 3)
        pose (torch.Tensor): (4, 4)
        depth_margin (float, optional): depth margin. Defaults to 0.05.
        
    Returns:
        proj_xyz (torch.Tensor): (N, 3) the projected points
        proj_front (torch.Tensor): (N,) True if the point is projected in front of the depth map
        proj_align (torch.Tensor): (N,) True if the point is projected within the margin the depth map
        proj_behind (torch.Tensor): (N,) True if the point is projected behind the depth map
    '''
    proj_xy, proj_z = project_points_world2plane(
        points, # (N, 3)
        pose.unsqueeze(0), # (1, 4, 4) 
        K, # (3, 3)
    ) # (1, N, 2), (1, N)
    proj_xy = proj_xy.reshape((-1, 2)) # (N, 2)
    proj_z = proj_z.reshape((-1)) # (N, )

    proj_xyz = torch.cat([
        proj_xy, proj_z.unsqueeze(-1)
    ], dim=-1) # (N, 3)
    
    proj_xy_within = torch.stack([
        0 <= proj_xy[:, 0], proj_xy[:, 0] < depth_tensor.shape[1], 
        0 <= proj_xy[:, 1], proj_xy[:, 1] < depth_tensor.shape[0]
    ], dim=-1) # (N, 2)
    proj_xy_within = torch.all(proj_xy_within, dim=-1) # (N, )
    proj_z_within = proj_z > 0
    proj_within = torch.logical_and(
        proj_xy_within, proj_z_within
    ) # (N, )
    
    # Initialize the projection masks
    proj_front = torch.full_like(proj_within, fill_value=False) # (N, )
    proj_align = torch.full_like(proj_within, fill_value=False) # (N, )
    proj_behind = torch.full_like(proj_within, fill_value=False) # (N, )
    
    if proj_within.any():
        depth_within = depth_tensor[
            proj_xy[proj_within][:, 1].long(),
            proj_xy[proj_within][:, 0].long()
        ] # (N, )
        proj_z_within = proj_z[proj_within] # (N, )
        
        proj_front[proj_within] = proj_z_within < depth_within - depth_margin # (N, )
        proj_align[proj_within] = torch.abs(proj_z_within - depth_within) <= depth_margin # (N, )
        proj_behind[proj_within] = proj_z_within > depth_within + depth_margin # (N, )
    
    return proj_xyz, proj_front, proj_align, proj_behind

    
def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    - R: A 3x3 rotation matrix.
    
    Returns:
    - A quaternion in the format [x, y, z, w].
    """
    # Make sure the matrix is a numpy array
    R = np.asarray(R)
    # Allocate space for the quaternion
    q = np.empty((4,), dtype=np.float32)
    # Compute the quaternion components
    q[3] = np.sqrt(np.maximum(0, 1 + R[0, 0] + R[1, 1] + R[2, 2])) / 2
    q[0] = np.sqrt(np.maximum(0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2
    q[1] = np.sqrt(np.maximum(0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2
    q[2] = np.sqrt(np.maximum(0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2
    q[0] *= np.sign(q[0] * (R[2, 1] - R[1, 2]))
    q[1] *= np.sign(q[1] * (R[0, 2] - R[2, 0]))
    q[2] *= np.sign(q[2] * (R[1, 0] - R[0, 1]))
    return q

def quaternion_to_rotation_matrix(q):
    """
    Convert a quaternion into a rotation matrix.
    
    Parameters:
    - q: A quaternion in the format [x, y, z, w].
    
    Returns:
    - A 3x3 rotation matrix.
    """
    w, x, y, z = q[3], q[0], q[1], q[2]
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])


def relative_transformation(
    trans_01: torch.Tensor, trans_02: torch.Tensor, orthogonal_rotations: bool = False
) -> torch.Tensor:
    r"""Function that computes the relative homogenous transformation from a
    reference transformation :math:`T_1^{0} = \begin{bmatrix} R_1 & t_1 \\
    \mathbf{0} & 1 \end{bmatrix}` to destination :math:`T_2^{0} =
    \begin{bmatrix} R_2 & t_2 \\ \mathbf{0} & 1 \end{bmatrix}`.

    .. note:: Works with imperfect (non-orthogonal) rotation matrices as well.

    The relative transformation is computed as follows:

    .. math::

        T_1^{2} = (T_0^{1})^{-1} \cdot T_0^{2}

    Arguments:
        trans_01 (torch.Tensor): reference transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        trans_02 (torch.Tensor): destination transformation tensor of shape
         :math:`(N, 4, 4)` or :math:`(4, 4)`.
        orthogonal_rotations (bool): If True, will invert `trans_01` assuming `trans_01[:, :3, :3]` are
            orthogonal rotation matrices (more efficient). Default: False

    Shape:
        - Output: :math:`(N, 4, 4)` or :math:`(4, 4)`.

    Returns:
        torch.Tensor: the relative transformation between the transformations.

    Example::
        >>> trans_01 = torch.eye(4)  # 4x4
        >>> trans_02 = torch.eye(4)  # 4x4
        >>> trans_12 = gradslam.geometry.geometryutils.relative_transformation(trans_01, trans_02)  # 4x4
    """
    if not torch.is_tensor(trans_01):
        raise TypeError(
            "Input trans_01 type is not a torch.Tensor. Got {}".format(type(trans_01))
        )
    if not torch.is_tensor(trans_02):
        raise TypeError(
            "Input trans_02 type is not a torch.Tensor. Got {}".format(type(trans_02))
        )
    if not trans_01.dim() in (2, 3) and trans_01.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_01.shape)
        )
    if not trans_02.dim() in (2, 3) and trans_02.shape[-2:] == (4, 4):
        raise ValueError(
            "Input must be a of the shape Nx4x4 or 4x4."
            " Got {}".format(trans_02.shape)
        )
    if not trans_01.dim() == trans_02.dim():
        raise ValueError(
            "Input number of dims must match. Got {} and {}".format(
                trans_01.dim(), trans_02.dim()
            )
        )
    trans_10: torch.Tensor = (
        inverse_transformation(trans_01)
        if orthogonal_rotations
        else torch.inverse(trans_01)
    )
    trans_12: torch.Tensor = compose_transformations(trans_10, trans_02)
    return trans_12