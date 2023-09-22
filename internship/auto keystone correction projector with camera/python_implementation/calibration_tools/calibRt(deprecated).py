import cv2
import numpy as np

"""
在已知两个内参的前提下，标定R和t的代码
如果确定用弧面，可以省去分割过程
一个平面无法完成标定，因为我们用到了本质矩阵，这个矩阵在特征点共面的情况下会退化，其解会受噪声干扰，无法得到正确的R和t
"""
# 读投影图，获得左右半扇
p = cv2.imread("cp.jpg")
pl = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY).copy()
pr = pl.copy()
pl[:, 960:] = 0
pr[:, :960] = 0
# 定义相机内参，投影仪内参和畸变系数
camera_mtx = np.array([[598.46979743, 0., 318.86661906],
                       [0., 598.30945874, 205.43676911],
                       [0., 0., 1., ]])
projector_mtx = np.array([[2304., 0., 960.], [0., 2304., 1080.], [0., 0., 1.]])
dist_coef = np.array([2.15362815e-01, - 1.10434423e+00, 4.45313876e-04, 4.08613255e-03,
                      8.52395384e-01])

Corners_c = np.zeros((0, 1, 2))
Corners_p = np.zeros((0, 1, 2))

# 读相机图，分割
camera_imgs = {"cc1.jpg": 346, "cc2.jpg": 340, "cc3.jpg": 356, "cc4.jpg": 415, "cc5.jpg": 320}

for name, segmentation in camera_imgs.items():
    c = cv2.imread(name)
    c = cv2.undistort(c, camera_mtx, dist_coef)
    cl = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY).copy()
    cr = c.copy()
    cl[:, segmentation:] = 0
    cr[:, :segmentation] = 0
    # 棋盘格内部的行数和列数
    pattern_size = (8, 9)
    # 寻找棋盘格角点
    retval_pl, corners_pl = cv2.findChessboardCorners(pl, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
    retval_pr, corners_pr = cv2.findChessboardCorners(pr, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
    retval_cl, corners_cl = cv2.findChessboardCorners(cl, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
    retval_cr, corners_cr = cv2.findChessboardCorners(cr, pattern_size, cv2.CALIB_CB_NORMALIZE_IMAGE)
    # 在图像上绘制角点
    cv2.drawChessboardCorners(p, pattern_size, corners_pl, retval_pl)
    cv2.drawChessboardCorners(p, pattern_size, corners_pr, retval_pr)
    cv2.drawChessboardCorners(c, pattern_size, corners_cl, retval_cl)
    cv2.drawChessboardCorners(c, pattern_size, corners_cr, retval_cr)
    # 展示角点识别结果
    cv2.imshow('Chessboard Corners', p)
    cv2.waitKey(100)
    cv2.imshow('Chessboard Corners', c)
    cv2.waitKey(100)
    cv2.destroyAllWindows()
    # 将左右半扇的角点拼一起，得到两个一一对应的特征点列表
    corners_p = np.concatenate((corners_pl, corners_pr), axis=0)
    corners_c = np.concatenate((corners_cl, corners_cr), axis=0)

    Corners_c = np.concatenate((Corners_c, corners_c), axis=0)
    Corners_p = np.concatenate((Corners_p, corners_p), axis=0)

# 将角点列表去掉一个多余的维度
Corners_c = np.squeeze(Corners_c, axis=1)
Corners_p = np.squeeze(Corners_p, axis=1)
# 在最后一列添加值为 1 的列，增广为齐次坐标
ones_column = np.ones((Corners_p.shape[0], 1))
corners_p_ = np.hstack((Corners_p, ones_column))
corners_c_ = np.hstack((Corners_c, ones_column))
# 归一化平面坐标
x_p_ = np.matrix(projector_mtx).I @ corners_p_.T
x_c_ = np.matrix(camera_mtx).I @ corners_c_.T
# 归一化坐标下，等效的内参，就是一个单位阵
K = np.eye(3)
# 求基础矩阵
fundamental_matrix, _ = cv2.findFundamentalMat(Corners_c, Corners_p, cv2.FM_LMEDS)
# 求本质矩阵
essential_matrix = projector_mtx.T @ fundamental_matrix @ camera_mtx
# 用opencv官方给的位姿恢复函数，从本质矩阵中恢复出R和t
# 因为opencv官方函数默认两个位姿下的相机内参是相同的，所以这里传入的是归一化坐标，用单位阵来等效两个一样的内参
# 这个函数会自动从本质矩阵的奇异值分解结果里面，通过传入的点坐标和内参，判断出四个可能解中的唯一正确解
_, R, t, __ = cv2.recoverPose(essential_matrix, x_c_[:2, :].T, x_p_[:2, :].T, K)  # 相机在投影仪坐标系下的描述
print("R=")
print(R)

rv = cv2.Rodrigues(R)[0]
print("rv=")
print(rv, np.linalg.norm(rv) / np.pi * 180)
print("t=")
print(t)
