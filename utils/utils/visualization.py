import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from sklearn.manifold import TSNE
from tqdm import tqdm

from utils.utils.open3d import (
    make_open3d_point_cloud,
    make_open3d_axes,
    make_open3d_corr_lines,
    make_open3d_corr_lines2,
    make_mesh_corr_lines,
    to_o3d_pcd
)

from experiments.lcrnet.modules.ops import point_to_node_partition
from experiments.lcrnet.modules.ops import apply_transform, inverse_transform

def vis_shifte_node(output_dict, vote_dict,src_masks=None, src_overlap=None,
        color=[0.7,0.7,0],
        src_node_color = [0.8, 0, 0],
        src_point_color = [0.8,0.4,0.4],
        ):
    
    src_node = output_dict['ori_anc_points_c']
    shifted_src_node = vote_dict['shifted_anc_points_c']
    src_points = output_dict['anc_points_f']
    centers = vote_dict['anc_points_c']
    src_node_knn_indices=output_dict['anc_node_knn_indices'][0]
    src_node_knn_masks=output_dict['anc_node_knn_masks'][0]
    # translation=[0,200,0]
    # translation2=[0,400,0]
    translation=[0,0,0]
    translation2=[0,0,0]
    # devise_AB=[0,-100,0]
    devise_BA=[0,0,0]
    ruler = o3d.geometry.TriangleMesh.create_sphere(radius=1)
    ruler = o3d.geometry.TriangleMesh.create_sphere(radius=0.15)
    # ruler.translate([150,0,0])
    ruler.paint_uniform_color([0, 0, 1])
    # ruler=[]

    radius = 0.2
    # radius = 0.01
    box_list1 = []
    box_list2 = []
    box_list3 = []
    if src_overlap is not None:
        src_color = src_overlap.detach().cpu().numpy()
    else:
        src_color = torch.ones(shifted_src_node.shape[0])*0.4

    
    for i in range(src_node.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(src_node[i].detach().reshape([3, 1]).cpu()).translate(devise_BA)
        # mesh_box.translate(translation)
        mesh_box.paint_uniform_color(src_node_color)

        mesh_box2 = o3d.geometry.TriangleMesh.create_sphere(radius=radius*0.5)
        mesh_box2.translate(shifted_src_node[i].detach().cpu().reshape([3, 1])).translate(devise_BA)
        # mesh_box2.translate(translation+translation2)
        mesh_box2.paint_uniform_color([src_color[i], 0, 0])

        mesh_box3 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box3.translate(src_node[i].detach().reshape([3, 1]).cpu()).translate(devise_BA)
        mesh_box3.translate(translation)
        mesh_box3.paint_uniform_color(src_node_color)

        mesh_box4 = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box4.translate(shifted_src_node[i].detach().cpu().reshape([3, 1])).translate(devise_BA)
        mesh_box4.translate(translation2)
        mesh_box4.paint_uniform_color(src_node_color)

        box_list1.append(mesh_box)
        # box_list1.append(mesh_box2)
        box_list2.append(mesh_box3)
        box_list3.append(mesh_box4)

    src_corr_lines = make_mesh_corr_lines(src_node.detach().cpu()+torch.Tensor(devise_BA), shifted_src_node.detach().cpu()+torch.Tensor(devise_BA),color)
    # src_corr_lines=*src_corr_lines
    # ref_corr_lines=*ref_corr_lines

    box_list4=[]
    if src_masks is not None:
        shifted_src_node=shifted_src_node[src_masks]

        for i in range(shifted_src_node.shape[0]):
            mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh_box.translate(shifted_src_node[i].detach().reshape([3, 1]).cpu()).translate(devise_BA)
            # mesh_box.translate(translation)
            mesh_box.paint_uniform_color(src_node_color)

            box_list4.append(mesh_box)
    elif centers is not None:
        for i in range(centers.shape[0]):
            mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            mesh_box.translate(centers[i].detach().reshape([3, 1]).cpu()).translate(devise_BA)
            # mesh_box.translate(translation)
            mesh_box.paint_uniform_color(src_node_color)

            box_list4.append(mesh_box)
        


    pcd0 = to_o3d_pcd(src_points.view(-1,3)).translate(devise_BA)
    # pcd01 = to_o3d_pcd(src_points.view(-1,3)).translate(translation).translate(devise_BA)
    # pcd02 = to_o3d_pcd(src_points.view(-1,3)).translate(translation2).translate(devise_BA)


    """draw partition"""
    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)]+box_list2)
    _, _, ori_src_node_knn_indices, ori_src_node_knn_masks = point_to_node_partition(
            src_points, src_node, 1024
        )
    ori_src_node_knn_indices[~ori_src_node_knn_masks]=0
    draw_point_to_node(src_points.cpu().detach().numpy(),src_node.cpu().detach().numpy(),ori_src_node_knn_indices.cpu().detach().numpy())


    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color),*src_corr_lines]+box_list1)


    o3d.visualization.draw_geometries([pcd0.paint_uniform_color(src_point_color)]+box_list4)
    _, _, src_node_knn_indices, src_node_knn_masks = point_to_node_partition(
            src_points, centers, 1024
        )
    src_node_knn_indices[~src_node_knn_masks]=0
    draw_point_to_node(src_points.cpu().detach().numpy(),centers.cpu().detach().numpy(),src_node_knn_indices.cpu().detach().numpy())


def visualization(
    output_dict,
    transform=None,
    src_points=None,
    ref_points=None,
    src_node_knn_indices=None,
    src_node_knn_masks=None,
    ref_node_knn_indices=None,
    ref_node_knn_masks=None,

    src_node_color = [1, 1, 0.1],
    ref_node_color = [1, 1, 0.1],
    src_point_color = [0.1, 0.3, 0.1],
    ref_point_color = [0.1, 0.1, 0.3],
    offsets=(0, 0, 30),
    find_true=False
    ):

    ref_corr_points=(output_dict['pos_corr_points'])
    src_corr_points=(output_dict['anc_corr_points'])
    ref_node = output_dict['pos_points_c']
    src_node = output_dict['anc_points_c']
    ref_node_corr_indices = output_dict['pos_node_corr_indices']
    src_node_corr_indices = output_dict['anc_node_corr_indices']

    ref_points=(output_dict['pos_points_f'])
    src_points=(output_dict['anc_points_f'])
    estimated_transform=(output_dict['estimated_transform'])


    src_node_knn_indices=output_dict['anc_node_knn_indices'][0]
    src_node_knn_masks=output_dict['anc_node_knn_masks'][0]
    ref_node_knn_indices=output_dict['pos_node_knn_indices'][0]
    ref_node_knn_masks=output_dict['pos_node_knn_masks'][0]

    if transform is not None:
        src_node = apply_transform(src_node, transform)
        src_points = apply_transform(src_points, transform)
        src_corr_points = apply_transform(src_corr_points, transform)


    # ref_node_knn_indices[~ref_node_knn_masks]=0
    # draw_point_to_node(ref_points.cpu(),ref_node.cpu(),ref_node_knn_indices.cpu())

    # ##### gt_node_corres
    # ref_node_knn_indices[~ref_node_knn_masks]=0
    # src_node_knn_indices[~src_node_knn_masks]=0
    # draw_node_correspondences(
    #     ref_points.cpu().numpy(),
    #     ref_node.cpu().numpy(),
    #     src_points.cpu().numpy(),
    #     src_node.cpu().numpy(),
    #     output_dict['gt_node_corr_indices'].cpu().numpy(),
    #     ref_node_knn_indices.cpu().numpy(),
    #     src_node_knn_indices.cpu().numpy(),
    #     )
    
    ##### predict_node_corres
    ref_node_knn_indices[~ref_node_knn_masks]=0
    src_node_knn_indices[~src_node_knn_masks]=0
    node_corr_indices = torch.stack([ref_node_corr_indices,src_node_corr_indices],dim=1)
    draw_node_correspondences(
        ref_points.cpu().numpy(),
        ref_node.cpu().numpy(),
        src_points.cpu().numpy(),
        src_node.cpu().numpy(),
        node_corr_indices.cpu().numpy(),
        ref_node_colors=ref_node_color,
        src_node_colors=src_node_color,
        ref_point_colors=ref_point_color,
        src_point_colors=src_point_color,
        offsets=offsets,
        )

    ref_color = np.ones(ref_node.shape)*0.4
    src_color = np.ones(src_node.shape)*0.4
    ref_color[:,0]=0.8
    src_color[:,2]=0.8

    if  ref_node_knn_indices is not None:
        ref_node_knn_indices[~ref_node_knn_masks]=0
        src_node_knn_indices[~src_node_knn_masks]=0
        ref_node_knn_indices=ref_node_knn_indices
        src_node_knn_indices=src_node_knn_indices


    #### vis node corr
    node_corr_indices = torch.stack([ref_node_corr_indices,src_node_corr_indices],dim=1)
    if find_true:
        true_corr, false_corr, preciseion= find_true_false_node(src_node,ref_node,node_corr_indices,output_dict['gt_node_corr_indices'])
        draw_node_correspondences(
            ref_points.cpu().numpy(),
            ref_node.cpu().numpy(),
            src_points.cpu().numpy(),
            src_node.cpu().numpy(),
            node_corr_indices.cpu().numpy(),
            ref_node_colors=ref_node_color,
            src_node_colors=src_node_color,
            ref_point_colors=ref_point_color,
            src_point_colors=src_point_color,
            offsets=offsets,
            true_corr=true_corr,
            false_corr=false_corr
            )

    #### vis point corr
    if find_true:
        true_mask = find_true_false(src_corr_points,ref_corr_points, transform)
        # print(true_mask.sum()/true_mask.shape[0])
        print(true_mask.sum(),'/',src_corr_points.size(0),'=',true_mask.sum()/src_corr_points.size(0))
    else:
        true_mask = None
    draw_point_correspondences(
        ref_points.cpu().numpy(),
        src_points.cpu().numpy(),
        src_corr_points.cpu().numpy(),
        ref_corr_points.cpu().numpy(),
        ref_point_colors=ref_point_color,
        src_point_colors=src_point_color,
        offsets=offsets,
        true_mask=true_mask
        )

    # if estimated_transform is not None:
    src_points_raw_t = apply_transform(src_points, estimated_transform)
    src_pcd = make_open3d_point_cloud(src_points_raw_t.cpu().numpy())
    ref_pcd = make_open3d_point_cloud(ref_points.cpu().numpy())
    o3d.visualization.draw_geometries([src_pcd.paint_uniform_color([0,1,0]), ref_pcd.paint_uniform_color([1,0,0])])

def find_true_false(src_corr_points, ref_corr_points, transform, node_corr_indices=None, thres=1):
    # src_node = apply_transform(src_node, transform)
    # src_points = apply_transform(src_points, transform)
    src_corr_points = apply_transform(src_corr_points, transform)
    if node_corr_indices is None:
        # src_corr_points = apply_transform(src_corr_points, transform)
        # true = torch.norm(src_corr_points-ref_corr_points,dim=-1)<thres
        corr_distances = torch.linalg.norm(ref_corr_points - src_corr_points, dim=1)
        true = torch.lt(corr_distances, thres)
    else:
        true = torch.norm(src_corr_points[node_corr_indices[:,1]]-ref_corr_points[node_corr_indices[:,0]],dim=-1)<thres
    return true

def find_true_false_node(src_corr_points, ref_corr_points, node_corr_indices, gt_corr):
    # Find the indices of the reference and source correspondences that are in the ground-truth correspondences
    ref_gt_corr_indices = gt_corr[:, 0]
    src_gt_corr_indices = gt_corr[:, 1]

    # Find the indices of the reference and source correspondences that are in the predicted correspondences
    ref_corr_indices = node_corr_indices[:, 0]
    src_corr_indices = node_corr_indices[:, 1]

    # Create a matrix where the rows are the reference correspondences and the columns are the source correspondences
    gt_corr_mat = torch.zeros((ref_corr_points.size()[0], src_corr_points.size()[0]))
    gt_corr_mat[ref_gt_corr_indices, src_gt_corr_indices] = 1.0

    pred_corr_mat = torch.zeros_like(gt_corr_mat)
    pred_corr_mat[ref_corr_indices, src_corr_indices] = 1.0

    pos_corr_mat = gt_corr_mat * pred_corr_mat
    true = pos_corr_mat==1
    false = (pred_corr_mat-pos_corr_mat)==1

    # Find the indices of the true and false correspondences
    true = true.numpy()
    false = false.numpy()
    true = np.argwhere( true==True )
    false = np.argwhere( false==True )

    # Print the precision
    num_gt_correspondences = gt_corr_mat.sum()
    num_pred_correspondences = pred_corr_mat.sum()
    num_pos_correspondences = pos_corr_mat.sum()
    print(num_pos_correspondences/num_pred_correspondences)

    return true, false, num_pos_correspondences/num_pred_correspondences

def draw_point_to_node(points, nodes, point_to_node, node_colors=None):
    if node_colors is None:
        node_colors = np.random.rand(*nodes.shape)
    # point_colors = node_colors[point_to_node] * make_scaling_along_axis(points, alpha=0.3).reshape(-1, 1)
    point_colors=np.zeros_like(points)
    for i in range(nodes.shape[0]):
        point_colors[point_to_node[i,:]] = node_colors[i]

    # node_colors = np.ones_like(nodes) * np.array([[1, 0, 0]])

    # ncd = make_open3d_point_cloud(nodes, colors=node_colors)

    radius=0.3
    box_list1=[]
    for i in range(nodes.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(nodes[i].reshape([3, 1]))
        # mesh_box.translate(translation)
        mesh_box.paint_uniform_color([1,0,0])

        box_list1.append(mesh_box)

    pcd = make_open3d_point_cloud(points, colors=point_colors)
    axes = make_open3d_axes()

    o3d.visualization.draw_geometries([pcd]+box_list1)


import torch
def draw_node_correspondences(
    ref_points,
    ref_nodes,
    src_points,
    src_nodes,
    node_correspondences,
    # ref_point_to_node=None,
    # src_point_to_node=None,
    ref_node_colors=None,
    src_node_colors=None,
    ref_point_colors=None,
    src_point_colors=None,
    offsets=(0, 0, 60),
    true_corr=None,
    false_corr=None
):

    src_nodes = src_nodes + offsets
    src_points = src_points + offsets

    if ref_node_colors is None:
        ref_node_colors=[1, 0, 0]
    if src_node_colors is None:
        src_node_colors=[0, 0, 1]

    radius=0.5
    box_list1=[]
    for i in range(src_nodes.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(src_nodes[i].reshape([3, 1]))
        # mesh_box.translate(translation)
        mesh_box.paint_uniform_color(src_node_colors)

        if true_corr is not None:
            if i in true_corr[:,1]:
                mesh_box.paint_uniform_color([1,0,0])
            elif i in false_corr[:,1]:
                mesh_box.paint_uniform_color([0,1,0])
            else:
                continue

        box_list1.append(mesh_box)
    
    box_list2=[]
    for i in range(ref_nodes.shape[0]):
        mesh_box = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        mesh_box.translate(ref_nodes[i].reshape([3, 1]))
        # mesh_box.translate(translation)
        mesh_box.paint_uniform_color(ref_node_colors)

        if true_corr is not None:
            if i in true_corr[:,0]:
                mesh_box.paint_uniform_color([0,1,0])
            elif i in false_corr[:,0]:
                mesh_box.paint_uniform_color([1,0,0])
            else:
                continue
        box_list2.append(mesh_box)

    # ref_ncd = make_open3d_point_cloud(ref_nodes, colors=ref_node_colors)
    ref_pcd = make_open3d_point_cloud(ref_points)
    # src_ncd = make_open3d_point_cloud(src_nodes, colors=src_node_colors)
    src_pcd = make_open3d_point_cloud(src_points)
    # corr_lines = make_open3d_corr_lines2(ref_nodes, src_nodes, node_correspondences)
    axes = make_open3d_axes(scale=0.1)

    if true_corr is None:
        corr_lines = make_mesh_corr_lines(torch.tensor(ref_nodes), torch.tensor(src_nodes),[0,1,0],0.1,corres=node_correspondences)
        o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), *corr_lines, axes]+box_list1+box_list2)
    else:

        # o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), axes]+box_list2)

        if (~true_corr).sum()==0:
            t_corr_lines = make_mesh_corr_lines(torch.tensor(ref_nodes), torch.tensor(src_nodes),[0,1,0],0.1,corres=true_corr)
            o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), *t_corr_lines, axes]+box_list1+box_list2)
        else:
            n_corr_lines = make_mesh_corr_lines(torch.tensor(ref_nodes), torch.tensor(src_nodes),[1,0,0],0.1,corres=false_corr)
            t_corr_lines = make_mesh_corr_lines(torch.tensor(ref_nodes), torch.tensor(src_nodes),[0,1,0],0.1,corres=true_corr)
            o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), *n_corr_lines, *t_corr_lines, axes]+box_list1+box_list2)


def draw_point_correspondences(
    ref_points,
    # ref_nodes,
    src_points,
    # src_nodes,
    src_corr_points,
    ref_corr_points,
    # ref_point_to_node=None,
    # src_point_to_node=None,
    # ref_node_colors=None,
    # src_node_colors=None,
    ref_point_colors=None,
    src_point_colors=None,
    offsets=(0, 0, 60),
    true_mask=None
):
    # src_nodes = src_nodes + offsets
    src_points = src_points + offsets
    src_corr_points = src_corr_points + offsets

    # if ref_node_colors is None:
    #     ref_node_colors = np.random.rand(*ref_nodes.shape)
    #     if ref_point_to_node is not None:
    #         ref_point_colors=np.zeros_like(ref_points)
    #         for i in range(ref_nodes.shape[0]):
    #             ref_point_colors[ref_point_to_node[i,:]] = ref_node_colors[i]
    #     ref_node_colors = np.ones_like(ref_nodes) * np.array([[1, 0, 0]])

    # if src_node_colors is None:
    #     src_node_colors = np.random.rand(*src_nodes.shape)
    #     if ref_point_to_node is not None:
    #         src_point_colors=np.zeros_like(src_points)
    #         for i in range(src_nodes.shape[0]):
    #             src_point_colors[src_point_to_node[i,:]] = src_node_colors[i]
    #     src_node_colors = np.ones_like(src_nodes) * np.array([[1, 0, 0]])\
    
    ref_pcd = make_open3d_point_cloud(ref_points)
    src_pcd = make_open3d_point_cloud(src_points)
    axes = make_open3d_axes(scale=0.1)

    if true_mask is not None:
        true_mask=true_mask.cpu().numpy()
        ture_src_corr_points = src_corr_points[true_mask]
        ture_ref_corr_points = ref_corr_points[true_mask]

        false_src_corr_points = src_corr_points[~true_mask]
        false_ref_corr_points = ref_corr_points[~true_mask]

        t_idx=np.arange(ture_src_corr_points.shape[0]).reshape(-1,1)
        t_point_correspondences=np.concatenate([t_idx,t_idx],axis=1)

        n_idx=np.arange(false_src_corr_points.shape[0]).reshape(-1,1)
        n_point_correspondences=np.concatenate([n_idx,n_idx],axis=1)

    
        t_corr_lines = make_open3d_corr_lines2(ture_ref_corr_points, ture_src_corr_points, t_point_correspondences, 'true')
        n_corr_lines = make_open3d_corr_lines2(false_ref_corr_points, false_src_corr_points, n_point_correspondences, 'false')


        # corr_lines = make_mesh_corr_lines(torch.tensor(ref_corr_points), torch.tensor(src_corr_points),[0,1,0],0.1)
        # o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors),  axes])
        o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), t_corr_lines, n_corr_lines, axes])

    else:
        idx=np.arange(ref_corr_points.shape[0]).reshape(-1,1)
        point_correspondences=np.concatenate([idx,idx],axis=1)

    
        corr_lines = make_open3d_corr_lines2(ref_corr_points, src_corr_points, point_correspondences)
        # corr_lines = make_mesh_corr_lines(torch.tensor(ref_corr_points), torch.tensor(src_corr_points),[0,1,0],0.1)
        # o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors),  axes])
        o3d.visualization.draw_geometries([ref_pcd.paint_uniform_color(ref_point_colors), src_pcd.paint_uniform_color(src_point_colors), corr_lines, axes])
