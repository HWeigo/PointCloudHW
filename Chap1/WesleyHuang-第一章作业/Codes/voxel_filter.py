# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d
import os
import numpy as np


#%%
# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
#     random_select: 若为True，使用随机选点，否则取中心
def voxel_filter(point_cloud, leaf_size, random_select=False):
    # 作业3
    # 屏蔽开始
    pcd = np.asarray(point_cloud.points)
    pcd_max = pcd.max(axis=0)
    pcd_min = pcd.min(axis=0)
    # Compute dimention of the voxel grid
    D = np.ceil((pcd_max - pcd_min) / leaf_size)
    # Compute voxel index for each point
    d = np.round((pcd - pcd_min) / leaf_size)

    # Perform: h = hx + hy * Dx + hz * Dx * Dy
    h_helper = np.array([[1], [D[0]], [D[0] * D[1]]])
    hs = np.dot(d, h_helper)

    # Store points within the same voxel into dictionary with same key
    dict = {}
    for idx, h in enumerate(hs):
        x = h[0]
        if x in dict.keys():
            dict[x] = np.vstack((dict[x], pcd[idx]))
        else:
            dict[x] = np.array([pcd[idx]])

    filtered_points = []
    # Iterate the dictionary. For each voxel, select the centroid / one point randomly.
    for v in dict.values():
        if v.shape[0] is 1:
            filtered_points.append(v[0])
        else:
            if random_select:
                random_idx = np.random.choice(v.shape[0], 1)
                filtered_points.append(v[random_idx][0])
            else:
                filtered_points.append(v.mean(axis=0))

    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

#%%
def main():
    # Load point cloud
    cat_index = 0
    root_dir = "../../modelnet40_normal_resampled/"
    models = os.listdir(root_dir)
    models = sorted(models)
    filename = os.path.join(root_dir, models[cat_index], models[cat_index] + '_0001.txt')

    pcd_array = np.loadtxt(filename, delimiter=',')[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    print("Number of point cloud: ", pcd_array.shape[0])

    # 调用voxel滤波函数，实现滤波
    filtered_cloud = voxel_filter(pcd, 0.2, random_select=False)
    pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([pcd])


if __name__ == '__main__':
    main()
