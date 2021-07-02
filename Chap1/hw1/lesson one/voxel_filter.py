# 实现voxel滤波，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
# from pyntcloud import PyntCloud

# 功能：对点云进行voxel滤波
# 输入：
#     point_cloud：输入点云
#     leaf_size: voxel尺寸
def voxel_filter(point_cloud, leaf_size, random_select = False):
    filtered_points = []
    # 作业3
    # 屏蔽开始
    pcd = np.asarray(point_cloud.points)
    pcd_max = pcd.max(axis=0)
    pcd_min = pcd.min(axis=0)
    D = np.ceil((pcd_max - pcd_min) / leaf_size)

    d = np.round((pcd - pcd_min) / leaf_size)

    # h = hx + hy * Dx + hz * Dx * Dy
    h_helper = np.array([[1],[D[0]],[D[0]*D[1]]])
    hs = np.dot(d, h_helper)

    dict = {}
    for idx, h in enumerate(hs):
        x = h[0]
        if x in dict.keys():
            dict[x] = np.vstack((dict[x], pcd[idx]))
        else:
            dict[x] = np.array([pcd[idx]])
    # print(dict)

    filtered_points = []
    for v in dict.values():
        print(v)
        if v.shape[0] is 1:
            filtered_points.append(v[0])
        else:
            if random_select:
                random_idx = np.random.choice(v.shape[0], 1)
                # print(v[random_idx, :])
                filtered_points.append(v[random_idx][0])
            else:
                filtered_points.append(v.mean(axis=0))



    # 屏蔽结束

    # 把点云格式改成array，并对外返回
    filtered_points = np.array(filtered_points, dtype=np.float64)
    return filtered_points

def main():
    # # 从ModelNet数据集文件夹中自动索引路径，加载点云
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 加载自己的点云文件
    # file_name = "/Users/renqian/Downloads/program/cloud_data/11.ply"
    # point_cloud_pynt = PyntCloud.from_file(file_name)

    # 转成open3d能识别的格式
    # point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)
    # o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

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
    filtered_cloud = voxel_filter(pcd, 0.05, random_select=True) #100.0
    # print(filtered_cloud)
    pcd.points = o3d.utility.Vector3dVector(filtered_cloud)
    # 显示滤波后的点云
    o3d.visualization.draw_geometries([pcd])

if __name__ == '__main__':
    main()
