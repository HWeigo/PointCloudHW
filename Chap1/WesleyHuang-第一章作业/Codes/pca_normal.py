#%%

import numpy as np
import open3d as o3d
import os


#%%

# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data):
    if type(data) is o3d.cpu.pybind.utility.Vector3dVector:
        pcd = np.asarray(data)
    else:
        pcd = data

    if pcd.ndim is not 3:
        pcd = np.transpose(pcd)

    # Normalization
    mean = pcd.mean(axis=1, keepdims=True)
    pcd = pcd - mean

    # Implement SVD
    H = np.dot(pcd, pcd.T)
    u, s, vt = np.linalg.svd(H, hermitian=True)

    ##or directly use SVD to decompose pcd matrix, u is the same as above
    #u, s, vt = np.linalg.svd(pcd, hermitian=False)

    eigenvalues = s
    eigenvectors = u

    return eigenvalues, eigenvectors

#%%

def main():
    # Load Point Cloud
    cat_index = 0
    root_dir = "../../modelnet40_normal_resampled/"
    models = os.listdir(root_dir)
    models = sorted(models)
    filename = os.path.join(root_dir, models[cat_index], models[cat_index] + '_0001.txt')

    pcd_array = np.loadtxt(filename, delimiter=',')[:, 0:3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_array)
    print("Number of point cloud: ", pcd_array.shape[0])

    w, v = PCA(pcd.points)
    ori_1st = v[:, 0]
    print("The main orientation of point cloud:", ori_1st)

    # Draw main orientation
    # Note 1st/2nd/3rd orientation is marked as red/green/blue
    ori_points = [v[:,0], v[:,0]*-1, v[:,1], v[:,1]*-1, v[:,2], v[:,2]*-1]
    ori_lines = [[0,1], [2,3], [4,5]]
    ori_colors = [[1,0,0], [0,1,0], [0,0,1]]
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(ori_points)
    line_set.lines = o3d.utility.Vector2iVector(ori_lines)
    line_set.colors = o3d.utility.Vector3dVector(ori_colors)

    # Visualization
    o3d.visualization.draw_geometries([pcd, line_set])

    # Build KD-Tree
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    normals = []
    # 作业2
    # 屏蔽开始
    neighbor_size = 10
    for i in range(pcd_array.shape[0]):
        # Get the neighbor points' index within neighbor_size
        _, idx, _ = pcd_tree.search_knn_vector_3d(pcd.points[i], neighbor_size)
        # Perform PCA
        _, v = PCA(np.asarray(pcd.points)[idx, :])
        normals.append(v[:,2])

    # 屏蔽结束

    normals = np.array(normals, dtype=np.float64)
    # Store normal vector into point cloud's normal property. Press "N" to visualize.
    pcd.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([pcd])


#%%

if __name__ == '__main__':
    main()

