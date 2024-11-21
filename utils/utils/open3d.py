import numpy as np
import open3d as o3d
import torch


def get_color(color_name):
    if color_name == 'red':
        return np.asarray([1.0, 0.0, 0.0])
    elif color_name == 'blue':
        return np.asarray([0.0, 0.0, 1.0])
    elif color_name == 'green':
        return np.asarray([0.0, 1.0, 0.0])
    elif color_name == 'yellow':
        return np.asarray([0.0, 1.0, 1.0])
    else:
        raise RuntimeError(f'Unsupported color: {color_name}.')


def make_scaling_along_axis(points, axis=2, alpha=0):
    if isinstance(axis, int):
        new_scaling_axis = np.zeros(3)
        new_scaling_axis[axis] = 1
        axis = new_scaling_axis
    if not isinstance(axis, np.ndarray):
        axis = np.asarray(axis)
    axis /= np.linalg.norm(axis)
    projections = np.matmul(points, axis)
    upper = np.amax(projections)
    lower = np.amin(projections)
    scales = 1 - ((projections - lower) / (upper - lower) * (1 - alpha) + alpha)
    return scales


def make_open3d_colors(points, base_color, scaling_axis=2, scaling_alpha=0):
    if not isinstance(base_color, np.ndarray):
        base_color = np.asarray(base_color)
    colors = np.ones_like(points) * base_color
    scales = make_scaling_along_axis(points, axis=scaling_axis, alpha=scaling_alpha)
    colors = colors * scales.reshape(-1, 1)
    return colors


def make_open3d_point_cloud(points, colors=None, normals=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)
    return pcd


def estimate_normals(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.estimate_normals()
    normals = np.asarray(pcd.normals)
    return normals


def voxel_downsample(points, voxel_size, normals=None):
    pcd = make_open3d_point_cloud(points, normals=normals)
    pcd = pcd.voxel_down_sample(voxel_size)
    points = np.asarray(pcd.points)
    if normals is not None:
        normals = np.asarray(pcd.normals)
        return points, normals
    else:
        return points


def make_open3d_registration_feature(data):
    r"""
    Make open3d registration features

    :param data: numpy.ndarray (N, C)
    :return feats: o3d.pipelines.registration.Feature
    """
    feats = o3d.pipelines.registration.Feature()
    feats.data = data.T
    return feats


def to_array(tensor):
    """
    Conver tensor to array
    """
    if(not isinstance(tensor,np.ndarray)):
        if(tensor.device == torch.device('cpu')):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor
    
def to_o3d_pcd(xyz):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(to_array(xyz))
    return pcd

def open3d_draw(*geometries):
    o3d.visualization.draw_geometries(geometries)


def registration_with_ransac_from_feats(
    src_points,
    ref_points,
    src_feats,
    ref_feats,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=50000,
    val_iterations=1000,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_feats = make_open3d_registration_feature(src_feats)
    ref_feats = make_open3d_registration_feature(ref_feats)

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src_pcd,
        ref_pcd,
        src_feats,
        ref_feats,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, val_iterations),
    )

    return result.transformation


def registration_with_ransac_from_correspondences(
    src_points,
    ref_points,
    correspondences=None,
    distance_threshold=0.05,
    ransac_n=3,
    num_iterations=10000,
):
    r"""
    Compute the transformation matrix from src_points to ref_points
    """
    src_pcd = make_open3d_point_cloud(src_points)
    ref_pcd = make_open3d_point_cloud(ref_points)

    if correspondences is None:
        indices = np.arange(src_points.shape[0])
        correspondences = np.stack([indices, indices], axis=1)
    correspondences = o3d.utility.Vector2iVector(correspondences)

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src_pcd,
        ref_pcd,
        correspondences,
        distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        ransac_n=ransac_n,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(num_iterations, num_iterations),
    )
    return result.transformation


def make_mesh_corr_lines(ref_corr_points, src_corr_points, color, radius=0.1, corres=None):
    num_correspondences = ref_corr_points.shape[0]
    corr_points = np.concatenate([ref_corr_points, src_corr_points], axis=0)
    if corres is not None:
        corr_indices = [(corres[i,0], corres[i,1] + num_correspondences) for i in range(corres.shape[0])]
    else:
        corr_indices = [(i, i + num_correspondences) for i in range(num_correspondences)]

    correspondence_mesh = LineMesh(corr_points, corr_indices, color, radius=radius)
    return correspondence_mesh.cylinder_segments

def make_open3d_corr_lines2(ref_corr_points, src_corr_points, corres, type=None):
    num_correspondences = ref_corr_points.shape[0]
    corr_points = np.concatenate([ref_corr_points, src_corr_points], axis=0)
    corr_indices = [(corres[i,0], corres[i,1] + num_correspondences) for i in range(corres.shape[0])]
    corr_lines = o3d.geometry.LineSet()
    corr_lines.points = o3d.utility.Vector3dVector(corr_points)
    corr_lines.lines = o3d.utility.Vector2iVector(corr_indices)
    if type is None or type=='true':
        corr_lines.paint_uniform_color(np.asarray([0.0, 1.0, 0.0]))
    elif type=='false':
        corr_lines.paint_uniform_color(np.asarray([1.0, 0.0, 0.0]))
    else:
        print('line type wrong')


    return corr_lines


def make_open3d_axis(axis_vector=None, origin=None, scale=1.0):
    if origin is None:
        origin = np.zeros((1, 3))
    if axis_vector is None:
        axis_vector = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axis_vector = axis_vector * scale
    axis_point = origin + axis_vector
    points = np.concatenate([origin, axis_point], axis=0)
    line = np.array([[0, 1]], dtype=np.long)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(line)
    axes.paint_uniform_color(get_color('red'))
    return axes


def make_open3d_axes(axis_vectors=None, origin=None, scale=1.0):
    if origin is None:
        origin = np.zeros((1, 3))
    if axis_vectors is None:
        axis_vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axis_vectors = axis_vectors * scale
    axis_points = origin + axis_vectors
    points = np.concatenate([origin, axis_points], axis=0)
    lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.long)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float)
    axes = o3d.geometry.LineSet()
    axes.points = o3d.utility.Vector3dVector(points)
    axes.lines = o3d.utility.Vector2iVector(lines)
    axes.colors = o3d.utility.Vector3dVector(colors)
    return axes


def make_open3d_corr_lines(ref_corr_points, src_corr_points, label):
    num_correspondences = ref_corr_points.shape[0]
    corr_points = np.concatenate([ref_corr_points, src_corr_points], axis=0)
    corr_indices = [(i, i + num_correspondences) for i in range(num_correspondences)]
    corr_lines = o3d.geometry.LineSet()
    corr_lines.points = o3d.utility.Vector3dVector(corr_points)
    corr_lines.lines = o3d.utility.Vector2iVector(corr_indices)
    if label == 'pos':
        corr_lines.paint_uniform_color(np.asarray([0.0, 1.0, 0.0]))
    elif label == 'neg':
        corr_lines.paint_uniform_color(np.asarray([1.0, 0.0, 0.0]))
    else:
        raise ValueError('Unsupported `label` {} for correspondences'.format(label))
    return corr_lines


"""Module which creates mesh lines from a line set
Open3D relies upon using glLineWidth to set line width on a LineSet
However, this method is now deprecated and not fully supporeted in newer OpenGL versions
See:
    Open3D Github Pull Request - https://github.com/intel-isl/Open3D/pull/738
    Other Framework Issues - https://github.com/openframeworks/openFrameworks/issues/3460

This module aims to solve this by converting a line into a triangular mesh (which has thickness)
The basic idea is to create a cylinder for each line segment, translate it, and then rotate it.

License: MIT

"""

def align_vector_to_another(a=np.array([0, 0, 1]), b=np.array([1, 0, 0])):
    """
    Aligns vector a to vector b with axis angle rotation
    """
    if np.array_equal(a, b):
        return None, None
    axis_ = np.cross(a, b)
    axis_ = axis_ / np.linalg.norm(axis_)
    angle = np.arccos(np.dot(a, b))

    return axis_, angle

def normalized(a, axis=-1, order=2):
    """Normalizes a numpy array of points"""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis), l2

class LineMesh(object):
    def __init__(self, points, lines=None, colors=[0, 1, 0], radius=0.15):
        """Creates a line represented as sequence of cylinder triangular meshes

        Arguments:
            points {ndarray} -- Numpy array of ponts Nx3.

        Keyword Arguments:
            lines {list[list] or None} -- List of point index pairs denoting line segments. If None, implicit lines from ordered pairwise points. (default: {None})
            colors {list} -- list of colors, or single color of the line (default: {[0, 1, 0]})
            radius {float} -- radius of cylinder (default: {0.15})
        """
        self.points = np.array(points)
        self.lines = np.array(
            lines) if lines is not None else self.lines_from_ordered_points(self.points)
        self.colors = np.array(colors)
        self.radius = radius
        self.cylinder_segments = []

        self.create_line_mesh()

    @staticmethod
    def lines_from_ordered_points(points):
        lines = [[i, i + 1] for i in range(0, points.shape[0] - 1, 1)]
        return np.array(lines)

    def create_line_mesh(self):
        first_points = self.points[self.lines[:, 0], :]
        second_points = self.points[self.lines[:, 1], :]
        line_segments = second_points - first_points
        line_segments_unit, line_lengths = normalized(line_segments)

        z_axis = np.array([0, 0, 1])
        # Create triangular mesh cylinder segments of line
        for i in range(line_segments_unit.shape[0]):
            line_segment = line_segments_unit[i, :]
            line_length = line_lengths[i]
            # get axis angle rotation to allign cylinder with line segment
            axis, angle = align_vector_to_another(z_axis, line_segment)
            # Get translation vector
            translation = first_points[i, :] + line_segment * line_length * 0.5
            # create cylinder and apply transformations
            cylinder_segment = o3d.geometry.TriangleMesh.create_cylinder(
                self.radius, line_length)
            cylinder_segment = cylinder_segment.translate(
                translation, relative=False)
            if axis is not None:
                axis_a = axis * angle
                cylinder_segment = cylinder_segment.rotate(
                    R=np.array(o3d.geometry.get_rotation_matrix_from_axis_angle(axis_a)))
                # cylinder_segment = cylinder_segment.rotate(
                #   axis_a, center=True, type=o3d.geometry.RotationType.AxisAngle)
            # color cylinder
            color = self.colors if self.colors.ndim == 1 else self.colors[i, :]
            cylinder_segment.paint_uniform_color(color)

            self.cylinder_segments.append(cylinder_segment)

    def add_line(self, vis):
        """Adds this line to the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.add_geometry(cylinder)

    def remove_line(self, vis):
        """Removes this line from the visualizer"""
        for cylinder in self.cylinder_segments:
            vis.remove_geometry(cylinder)


