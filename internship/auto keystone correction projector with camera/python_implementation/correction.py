import cv2
import numpy as np
import apriltag
import matplotlib.pyplot as plt
from math import asin

"""
规定坐标系:
图像坐标系，光机坐标系，相机坐标系：传统
世界坐标系：原点和光机坐标系原点重合。光机坐标系按照重力方向反着转回去，z轴，x轴平行于地面，y轴沿重力向下
墙面坐标系：墙上的2D坐标系，y轴沿重力向下
下标：p代表光机，c代表相机，w代表世界，wall代表墙
"""


class KeystoneCorrection:
    def __init__(self, camera_intrinsic_mtx, camera_distortion,
                 projector_intrinsic_mtx, projector_distortion, projector_reso,
                 r_cinp, t_cinp, gravity_vector_standard, gravity_vector_using,
                 img_camera, img_projector,
                 ifbiggest):
        """
        :param camera_intrinsic_mtx: 相机内参 3x3
        :param camera_distortion: 相机畸变 1x5
        :param projector_intrinsic_mtx: 投影仪内参 3x3
        :param projector_distortion: 投影仪畸变 1x5
        :param projector_reso: 投影仪分辨率 元组
        :param r_cinp: 旋转矩阵 相机在光机坐标系下的描述
        :param t_cinp: 平移向量 相机在光机坐标系下的描述
        :param gravity_vector_standard: 标准水平状态下的加速度计值，要转换成光机坐标系，元组
        :param img_camera: 照片，opencv ndarray
        :param img_projector: 光机投射的画面 opencv ndarray
        :param ifbiggest: 是否要最大的。True为最大，False为最清晰
        """
        # 内参和畸变
        self.Kc = camera_intrinsic_mtx
        self.Kp = projector_intrinsic_mtx
        self.Dp = projector_distortion
        self.Dc = camera_distortion

        self.resolution_p = projector_reso
        self.aspect_ratio = self.resolution_p[0] / self.resolution_p[1]
        # 外参，重力
        self.R_cinp = r_cinp  # 相机在光机中的描述
        self.t_cinp = t_cinp  # 相机在光机中的描述
        self.gravity_vector_p_standard = np.array(gravity_vector_standard)  # 水平状态的加速度计值，光机坐标系
        self.gravity_vector_p = np.array(gravity_vector_using)  # 此时光机坐标系下的加速度计值
        # 画面和照片
        self.img_c = cv2.undistort(img_camera, self.Kc, self.Dc)
        self.img_p = cv2.undistort(img_projector, self.Kp, self.Dp)
        # self.img_p = img_projector
        # 矫正模式,True代表最大，False代表最清晰
        self.ifbiggest = ifbiggest
        self.plot_wall = None
        self.kps_p = None
        self.kps_xy_wall = None
        self.pose_wall_in_p = None

        self.offsets = None  # 四个角点的像素offsets
        self.rpy = None  # 光机在世界中的欧拉角
        self.corrected_corners_p = None
        self.__correct()
        self.__get_pose()

    def update(self, img_camera, gravity_vettor_using):
        """
        调用此函数来更新矫正信息。传入新的重力，相机图片，自动更新矫正信息和pose信息
        :param img_camera:
        :param gravity_vettor_using:
        :return:
        """
        self.img_c = cv2.undistort(img_camera, self.Kc, self.Dc)
        self.gravity_vector_p = gravity_vettor_using
        self.__correct()
        self.__get_pose()

    @staticmethod
    def __get_keypoints(img) -> np.ndarray:
        """
        获得图像上的特征点 包括棋盘格和apriltag
        :param img: 图像
        :return: N*2的array
        """

        def draw_kps() -> None:
            """
            绘制特征点
            """
            image_bgr = cv2.drawChessboardCorners(img, pattern_size, corners, True)
            for r in results:
                if r.tag_id != 26 and r.tag_id != 27 and r.tag_id != 28 and r.tag_id != 29:
                    continue
                for j in r.corners:
                    image_bgr = cv2.circle(image_bgr, np.array(j, dtype="int"), 3, (0, 255, 0), 3)
                image_bgr = cv2.circle(image_bgr, np.array(r.center, dtype="int"), 3, (0, 0, 255), 3)

            plt.imshow(image_bgr)
            plt.show()
            return

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 寻找棋盘格角点
        pattern_size = (8, 6)
        retval, corners = cv2.findChessboardCorners(img_gray, pattern_size, cv2.CALIB_CB_ADAPTIVE_THRESH)
        kps = np.squeeze(corners, axis=1)

        # 再找apriltag
        # 创建AprilTag检测器
        options = apriltag.DetectorOptions(families='tag16h5')  # 选择AprilTag类型
        detector = apriltag.Detector(options)
        # 进行AprilTag检测
        results = detector.detect(img_gray)
        results = sorted(results, key=lambda x: x.tag_id)
        for i in results:
            if i.tag_id != 26 and i.tag_id != 27 and i.tag_id != 28 and i.tag_id != 29:
                continue
            kps = np.vstack((kps, i.corners))
            kps = np.vstack((kps, np.expand_dims(i.center, axis=0)))

        draw_kps()
        return kps

    @staticmethod
    def __get_plane(kps_xyz) -> list:
        """
        拟合平面
        :param kps_xyz: 三维的点坐标，Nx3
        :return: 平面参数 ax+by+c = z   a，b，c
        """

        def plot3D() -> None:
            """
            绘制三维特征点+拟合的平面
            :return: None
            """
            # 创建一个新的三维图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 使用Axes3D类
            # 提取坐标分量
            x = kps_xyz[:, 0]
            y = kps_xyz[:, 1]
            z = kps_xyz[:, 2]
            # 绘制三维散点图
            ax.scatter(x, y, z, c='b', marker='o')  # c为颜色，marker为标记类型

            # 绘制拟合的平面
            xx, yy = np.meshgrid(np.linspace(min(x), max(x), 10), np.linspace(min(y), max(y), 10))
            zz = X[0] * xx + X[1] * yy + X[2]
            ax.plot_surface(xx, yy, zz, color='r', alpha=0.3)

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # 显示图形
            plt.show()

        # kps_xyz 的每一行表示一个点的 (x, y, z) 坐标
        # 构建系数矩阵 A 和结果向量 b
        A_ = np.hstack((kps_xyz[:, :2], np.ones((kps_xyz.shape[0], 1))))
        b = kps_xyz[:, 2]

        # 使用最小二乘法公式计算系数向量 x = (a, b, c)
        X = np.linalg.lstsq(A_, b, rcond=None)[0]

        # 提取拟合的系数
        a, b, c = X

        # 输出拟合的平面方程系数
        print(f"拟合的平面方程：{a:.4f}x + {b:.4f}y + {c:.4f} = z")

        plot3D()
        return X

    @staticmethod
    def __get_H(kps_from, kps_to) -> np.ndarray:
        """
        获取两组点的单应性矩阵
        :param kps_from: points_src
        :param kps_to: points_dst
        :return: 3x3单应性矩阵
        """
        H = cv2.findHomography(kps_from, kps_to)[0]
        print("单应性矩阵：\n")
        print(H)
        return H

    def __get_inscribe_rect(self, corners_wall) -> np.ndarray:
        """
        得到内接矩形
        :param corners_wall: 墙上坐标的四个角点，右x下y
        :return:  内接矩形的四个角点的墙上坐标，从左上开始，顺时针顺序
        """

        def plot_wall() -> None:
            """
            画墙上的模拟图
            :return: None
            """
            plt.axis('equal')
            plt.scatter(corners_wall[:, 0], -corners_wall[:, 1])
            plt.scatter(inscribe_rect_corners[:, 0], -inscribe_rect_corners[:, 1])
            plt.text(0, 0, "y axis is flipped")
            plt.show()

        # 注意提升数值稳定性，做异常处理
        def slope(point1, point2) -> float:
            """
            计算两点连线斜率
            :param point1:
            :param point2:
            :return:
            """
            try:
                vec = point2 - point1
                return vec[1] / vec[0]
            except ValueError:
                raise ValueError

        def distance(point1, point2) -> float:
            """
            计算两点距离
            :param point1:
            :param point2:
            :return:
            """
            return np.linalg.norm(point2 - point1)

        def point_in_line_by_x(point1, point2, xx) -> np.ndarray:
            """
            给定两个点和x，找到两点连线上横坐标为x的点
            :param point1:
            :param point2:
            :param xx:
            :return:
            """
            y__ = point1[1] + (xx - point1[0]) * (point2[1] - point1[1]) / (point2[0] - point1[0])
            return np.array([xx, y__])

        def point_in_line_by_y(point1, point2, yy) -> np.ndarray:
            """
            给定两个点和y，找到两点连线上横坐标为y的点
            :param point1:
            :param point2:
            :param yy:
            :return:
            """
            x__ = point1[0] + (yy - point1[1]) * (point2[0] - point1[0]) / (point2[1] - point1[1])
            return np.array([x__, yy])

        def point_in_line_by_aspect_ratio(point1, point2, launch_point, aspect_ratio) -> np.ndarray:
            """
            从launch_point按照横纵比射出一条线，与p1，p2连线的交点
            :param point1:
            :param point2:
            :param launch_point:
            :param aspect_ratio:
            :return:
            """
            k1 = slope(point2, point1)
            k2 = -1 / aspect_ratio
            b1 = point1[1] - k1 * point1[0]
            b2 = launch_point[1] - k2 * launch_point[0]
            x__ = (b2 - b1) / (k1 - k2)
            y__ = (k1 * b2 - k2 * b1) / (k1 - k2)
            return np.array([x__, y__])

        def intersection(point1, k1, point2, k2) -> np.ndarray:
            """
            返回两条直线的交点
            :param point1:
            :param k1:
            :param point2:
            :param k2:
            :return:
            """
            x_ = (k1 * point1[0] - k2 * point2[0] + point2[1] - point1[1]) / (k1 - k2)
            y_ = k1 * x_ + point1[1] - k1 * point1[0]
            return np.array([x_, y_])

        # 这个函数将所有的寻找内接矩形的任务分成了四种等效的情况。
        # 如果该任意四边形并不直接符合四种情况之一，则可以通过水平和竖直翻转来使其符合
        # 在返回值时，只需要将最终的结果翻转回去即可
        inscribe_rect_corners = np.zeros((4, 2))
        # 先颠倒一下y轴，符合正常数学坐标系，A左上角，B在右上，C在右下，D在左下, 向上为y正
        corners_wall[:, 1] = -corners_wall[:, 1]

        # 在现在的坐标系下面讨论
        flip_horizontal = False
        flip_vertical = False
        if ((distance(corners_wall[1], corners_wall[2]) > distance(corners_wall[0], corners_wall[3])) and self.ifbiggest) \
                or ((distance(corners_wall[1], corners_wall[2]) < distance(corners_wall[0], corners_wall[3])) and not self.ifbiggest):
            # 水平翻转
            flip_horizontal = True
            corners_wall[:, 0] = -corners_wall[:, 0]
            corners_wall = np.array([corners_wall[1], corners_wall[0], corners_wall[3], corners_wall[2]])

        count = 0
        while count < 2:
            # 得到上，左，下的斜率
            A_, B_, C_, D_ = np.array(corners_wall)
            k = [slope(A_, B_), slope(D_, A_), slope(D_, C_), ]
            # 找左边的内接
            # 判断上左右边的斜率
            if k[0] > 0 and k[1] > 0 and k[2] > 0:
                # "PP 对角线"情形，对应000，111（需上下翻转）
                # 左上角为PP点，找右下对角线
                inscribe_rect_corners[0] = A_
                inscribe_rect_corners[2] = point_in_line_by_aspect_ratio(D_, C_, A_, self.aspect_ratio)
                inscribe_rect_corners[1] = np.array([inscribe_rect_corners[2][0], inscribe_rect_corners[0][1]])
                inscribe_rect_corners[3] = np.array([inscribe_rect_corners[0][0], inscribe_rect_corners[2][1]])
                break
            elif k[0] > 0 > k[2] and k[1] > 0:
                # "PP L"情形，对应001，011（需上下翻转）
                # 左上角为PP点，找竖直向下线
                inscribe_rect_corners[0] = A_
                inscribe_rect_corners[3] = point_in_line_by_x(D_, C_, A_[0])
                h = abs(inscribe_rect_corners[0][1] - inscribe_rect_corners[3][1])
                inscribe_rect_corners[1] = np.array([A_[0] + self.aspect_ratio * h, A_[1]])
                inscribe_rect_corners[2] = np.array([inscribe_rect_corners[1][0], inscribe_rect_corners[3][1]])
                break
            elif k[0] > 0 > k[1] and k[2] > 0:
                # 三点接触情形1，对应010，101（需上下翻转）
                # 左边找点，水平向右与底边交，竖直向上与上边交
                x = ((self.aspect_ratio * (k[1] - k[0]) * A_[0]) - ((1 / k[2] - 1 / k[1]) * (A_[1] - D_[1] - k[1] * A_[0]))) / (
                        (1 / k[2] - 1 / k[1]) * k[1] - self.aspect_ratio * (k[0] - k[1]))

                y = point_in_line_by_x(A_, D_, x)[1]

                w = (1 / k[2] - 1 / k[1]) * (y - D_[1])
                h = (k[0] - k[1]) * (x - A_[0])
                inscribe_rect_corners[0] = np.array([x, y + h])
                inscribe_rect_corners[1] = np.array([x + w, y + h])
                inscribe_rect_corners[2] = np.array([x + w, y])
                inscribe_rect_corners[3] = np.array([x, y])

                break
            elif k[0] < 0 < k[1] and k[2] > 0:
                # 三点接触情形2，对应100，110（需上下翻转）
                # 上边找点，水平向左与左边交，竖直向下与下边交，保证两个长度为长宽比
                xi = intersection(A_, k[0], D_, k[2])[0]
                x = ((1 / k[1] - 1 / k[0]) * k[0] * A_[0] + self.aspect_ratio * (k[2] - k[0]) * xi) / (
                        (self.aspect_ratio * (k[2] - k[0])) + (1 / k[1] - 1 / k[0]) * k[0])

                y = point_in_line_by_x(A_, B_, x)[1]

                w = x - point_in_line_by_y(A_, D_, y)[0]
                h = y - point_in_line_by_x(D_, C_, x)[1]

                inscribe_rect_corners[0] = np.array([x - w, y])
                inscribe_rect_corners[1] = np.array([x, y])
                inscribe_rect_corners[2] = np.array([x, y - h])
                inscribe_rect_corners[3] = np.array([x - w, y - h])

            # 跑到这里说明不符合上面四种情况，需要上下翻转
            flip_vertical = True
            corners_wall[:, 1] = -corners_wall[:, 1]
            corners_wall = np.array([corners_wall[3], corners_wall[2], corners_wall[1], corners_wall[0]])
            count += 1
            # 最好写一个应急返回值，一般不会发生，以防万一
            # 如果count==2，并且跑到了这里，说明程序出问题了，没找到，需要返回一个应急的结果
            if count == 2:
                raise "FindInscribeRectangleError"

        if flip_vertical:
            # 将y翻转回去
            corners_wall[:, 1] = -corners_wall[:, 1]
            corners_wall = np.array([corners_wall[3], corners_wall[2], corners_wall[1], corners_wall[0]])
            inscribe_rect_corners[:, 1] = -inscribe_rect_corners[:, 1]
            inscribe_rect_corners = np.array([inscribe_rect_corners[3], inscribe_rect_corners[2], inscribe_rect_corners[1], inscribe_rect_corners[0]])

        if flip_horizontal:
            # 将x翻转回去
            corners_wall[:, 0] = -corners_wall[:, 0]
            corners_wall = np.array([corners_wall[1], corners_wall[0], corners_wall[3], corners_wall[2]])
            inscribe_rect_corners[:, 0] = -inscribe_rect_corners[:, 0]
            inscribe_rect_corners = np.array([inscribe_rect_corners[1], inscribe_rect_corners[0], inscribe_rect_corners[3], inscribe_rect_corners[2]])

        corners_wall[:, 1] = -corners_wall[:, 1]
        inscribe_rect_corners[:, 1] = -inscribe_rect_corners[:, 1]
        plot_wall()
        return inscribe_rect_corners

    def __get_corrected_corners_p(self, inside_rect_wall, H_wall2p) -> np.ndarray:
        """
        得到矫正后的图像上的四个角点
        返回 4*2的array
        :param inside_rect_wall: 内接矩形的墙面坐标
        :param H_wall2p: 墙面到光机画面的单应性
        :return: 光机画面中的内接矩形角点坐标，也就是最终画面的显示范围
        """

        def shrink(rect) -> np.ndarray:
            """
            缩小rect的范围
            :param rect:
            :return:
            """
            A_, B_, C_, D_ = rect
            w = abs(B_[0] - A_[0])
            h = abs(A_[1] - D_[1])
            center = 0.25 * (A_ + B_ + C_ + D_)

            rect_shrunken = np.zeros((4, 2))
            rect_shrunken[0] = np.array([center[0] - 0.49 * w, center[1] - 0.49 * h])
            rect_shrunken[1] = np.array([center[0] + 0.49 * w, center[1] - 0.49 * h])
            rect_shrunken[2] = np.array([center[0] + 0.49 * w, center[1] + 0.49 * h])
            rect_shrunken[3] = np.array([center[0] - 0.49 * w, center[1] + 0.49 * h])
            return rect_shrunken

        # 计算一下新的投影范围，如果超出，则缩小内接矩形再算一次，直到符合范围要求
        in_range = False
        corrected_corners_p = None
        while not in_range:
            inside_rect_wall_ = np.hstack((inside_rect_wall, np.ones((4, 1))))
            corrected_corners_p_ = H_wall2p @ inside_rect_wall_.T
            corrected_corners_p = (corrected_corners_p_[:2, :] / corrected_corners_p_[2, :]).T
            in_range = True
            for point in corrected_corners_p:
                if not ((0 < point[0] < self.resolution_p[0]) and (0 < point[1] < self.resolution_p[1])):
                    in_range = False
                    break

            if not in_range:
                inside_rect_wall = shrink(inside_rect_wall)

        return corrected_corners_p

    def __get_pose(self) -> None:
        """
        :return: 墙坐标在光机坐标中的描述：旋转向量，平移向量
        """
        obj_points = np.hstack((self.kps_xy_wall, np.zeros((self.kps_xy_wall.shape[0], 1))))
        retval, rvec, tvec = cv2.solvePnP(obj_points, self.kps_p, self.Kp, self.Dp, cv2.SOLVEPNP_ITERATIVE)
        self.pose_wall_in_p = [rvec, tvec]

    def __get_kps_xyz_in_proj(self, kps_p, kps_c) -> np.ndarray:
        """
        得到关键点在光机坐标系中的三维坐标
        :param kps_p: 关键点的投影画面坐标
        :param kps_c: 关键点的照片画面坐标
        :return: 关键点的光机坐标
        """

        def plot3D() -> None:
            # 创建一个新的三维图形
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')  # 使用Axes3D类
            # 提取坐标分量
            x = kps_xyz_p[:, 0]
            y = kps_xyz_p[:, 1]
            z = kps_xyz_p[:, 2]
            # 绘制三维散点图
            ax.scatter(x, y, z, c='b', marker='o')  # c为颜色，marker为标记类型

            # 设置坐标轴标签
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            # 显示图形
            plt.show()

        # 求【R t
        #    0 1】 的逆，得到光机在相机中的描述，也就是相机的外参
        Rt = np.hstack((self.R_cinp, self.t_cinp))  # R和t左右拼接
        Rt01 = np.vstack((Rt, np.array([0, 0, 0, 1])))  # 下方再拼接一个 0 0 0 1
        Rt_I = np.matrix(Rt01).I  # 求逆
        # 相机的投影矩阵，相机内参x相机外参
        projection_matrix_c = self.Kc @ Rt_I[:3, :]
        # 光机的投影矩阵，即光机内参（增广）
        projection_matrix_p = np.hstack((self.Kp, np.zeros((3, 1))))
        # 进行三角测量，注意这里接受的array是2*n的
        kps_xyz = cv2.triangulatePoints(projection_matrix_p,
                                        projection_matrix_c,
                                        kps_p.T,
                                        kps_c.T
                                        ).T
        # 将齐次坐标转换为三维坐标
        kps_xyz_p = kps_xyz[:, :3] / kps_xyz[:, 3:]
        # 返回三维特征点的坐标（光机坐标系下）
        plot3D()
        return kps_xyz_p

    def __get_corners_wall(self, H_wall2p) -> np.ndarray:
        """
        得到墙上的四个角点坐标
        :param H_wall2p: 墙面到光机画面的单应性
        :return: 4*2array
        """

        def plot_wall() -> None:
            plt.scatter(self.plot_wall[:-4, 0], -self.plot_wall[:-4, 1])
            plt.scatter(self.plot_wall[-4:, 0], -self.plot_wall[-4:, 1])

        corners_p_ = np.array([[0, 0, 1],
                               [self.resolution_p[0], 0, 1],
                               [self.resolution_p[0], self.resolution_p[1], 1],
                               [0, self.resolution_p[1], 1]])
        corners_wall_ = np.matrix(H_wall2p).I @ corners_p_.T
        corners_wall = (corners_wall_[:2, :] / corners_wall_[2, :]).T
        print("四个角点的墙上坐标：")
        print(corners_wall)

        self.plot_wall = np.array(np.vstack((self.plot_wall, corners_wall)))
        plot_wall()

        return np.array(corners_wall)

    def __proj2world(self, kps_xyz_in_p) -> np.ndarray:
        """
        将关键点在光机坐标系转换到世界坐标系
        :param kps_xyz_in_p: 光机坐标系中的关键点
        :return: 关键点的世界坐标
        """
        G_cross = np.cross(self.gravity_vector_p, self.gravity_vector_p_standard)
        theta = asin(np.linalg.norm(G_cross) / (np.linalg.norm(self.gravity_vector_p) * np.linalg.norm(self.gravity_vector_p_standard)))
        rotation_vector_p2w = G_cross * (theta / np.linalg.norm(G_cross))
        R_p2w = cv2.Rodrigues(rotation_vector_p2w)[0]
        kps_xyz_w = (R_p2w @ kps_xyz_in_p.T).T
        return kps_xyz_w

    def __world2wall(self, kps_xyz_in_w, plane_coefs) -> np.ndarray:
        """
        将世界坐标系中的三维特征点转换到墙面上的二维点
        返回N*2的array
        :param kps_xyz_in_w: 关键点的世界坐标
        :param plane_coefs: 平面参数
        :return: 关键点的墙面坐标
        """
        # 获取平面系数 ax + by + d = z
        abc = np.array([plane_coefs[0], plane_coefs[1], -1])

        n_norm = np.linalg.norm(abc)
        n_wall = np.array([0, 0, -n_norm])
        n_cross = np.cross(abc, n_wall)
        theta = asin(np.linalg.norm(n_cross) / (n_norm ** 2))
        rotation_vector_w2wall = n_cross * (theta / np.linalg.norm(n_cross))
        R_w2wall = cv2.Rodrigues(rotation_vector_w2wall)[0]

        kps_xy_wall = (R_w2wall @ kps_xyz_in_w.T).T
        kps_xy_wall = kps_xy_wall[:, :2]
        print("墙上投影点坐标：")
        print(kps_xy_wall)
        # 重整坐标，保证左上角为0，0
        # kps_xy_wall -= np.min(kps_xy_wall, axis=0)

        self.plot_wall = kps_xy_wall

        return kps_xy_wall

    def __correct(self) -> None:
        """
        矫正环节，计算offsets
        :return:
        """

        def draw_corrected_area() -> None:
            """
            绘制矫正后的proj画面显示区域
            :return:
            """
            img = self.img_p.copy()
            img = cv2.line(img, (int(self.corrected_corners_p[0][0]), int(self.corrected_corners_p[0][1])),
                           (int(self.corrected_corners_p[1][0]), int(self.corrected_corners_p[1][1])), (0, 0, 255), 5)
            img = cv2.line(img, (int(self.corrected_corners_p[1][0]), int(self.corrected_corners_p[1][1])),
                           (int(self.corrected_corners_p[2][0]), int(self.corrected_corners_p[2][1])), (0, 0, 255), 5)
            img = cv2.line(img, (int(self.corrected_corners_p[2][0]), int(self.corrected_corners_p[2][1])),
                           (int(self.corrected_corners_p[3][0]), int(self.corrected_corners_p[3][1])), (0, 0, 255), 5)
            img = cv2.line(img, (int(self.corrected_corners_p[3][0]), int(self.corrected_corners_p[3][1])),
                           (int(self.corrected_corners_p[0][0]), int(self.corrected_corners_p[0][1])), (0, 0, 255), 5)
            plt.imshow(img)
            plt.show()

        # 得到画面和图像中的特征点，一一对应，N*2的array
        self.kps_p = self.__get_keypoints(self.img_p)
        kps_c = self.__get_keypoints(self.img_c)
        # 得到光机坐标系下的特征点三维坐标 N*3的array
        kps_xyz_p = self.__get_kps_xyz_in_proj(self.kps_p, kps_c)
        # 得到世界坐标系下的特征点三维坐标 N*3的array
        kps_xyz_w = self.__proj2world(kps_xyz_p)
        # 拟合平面，得到平面方程的参数
        plane_coefs = self.__get_plane(kps_xyz_w)
        self.kps_xy_wall = self.__world2wall(kps_xyz_w, plane_coefs)
        # 获得墙面到投影画面的单应性矩阵
        H_wall2p = self.__get_H(self.kps_xy_wall, self.kps_p)
        # 由单应性矩阵反推四个墙上角点
        corners_wall = self.__get_corners_wall(H_wall2p)
        # 获得内接矩形
        inside_rect_wall = self.__get_inscribe_rect(corners_wall)
        # 获得画面中的矫正后的角点
        self.corrected_corners_p = np.array(self.__get_corrected_corners_p(inside_rect_wall, H_wall2p), dtype="int")

        print("矫正后的角点在画面中：")
        print(self.corrected_corners_p)

        # 算offsets
        self.offsets = self.corrected_corners_p - np.array([[0, 0],
                                                            [self.resolution_p[0], 0],
                                                            [self.resolution_p[0], self.resolution_p[1]],
                                                            [0, self.resolution_p[1]]])

        draw_corrected_area()
        return


def get_adb_command(offsets) -> None:
    """
    根据offsets生成adb命令
    :return:
    """
    offsets = np.array(offsets, dtype="int")
    A = offsets[3]
    B = offsets[2]
    C = offsets[1]
    D = offsets[0]
    res = "adb shell setprop persist.vendor.hwc.keystone " + str(0 + abs(A[0])) + "," + str(0 + abs(A[1])) + "," + str(
        1920 - abs(B[0])) + "," + str(0 + abs(B[1])) + "," + str(1920 - abs(C[0])) + "," + str(
        1080 - abs(C[1])) + "," + str(0 + abs(D[0])) + "," + str(1080 - abs(D[1]))

    print(str(res))


if __name__ == "__main__":
    # 内参，畸变
    camera_mtx = np.array([[600.32380483, 0., 323.20302011],
                           [0., 600.39865037, 202.07122001],
                           [0., 0., 1.]])
    camera_dist = np.array([[2.48974585e-01, -1.49624210e+00, -4.27611936e-04, 5.42105284e-03,
                             1.84425912e+00]])

    projector_mtx = np.array([[2.28382693e+03, 0.00000000e+00, 9.89334268e+02],
                              [0.00000000e+00, 2.28805040e+03, 1.07939241e+03],
                              [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    projector_dist = np.array([[0.03103913, 0.02877732, -0.00038886, 0.00115072, -0.31241419]])

    projector_resolution = (1920, 1080)
    # 相机在投影仪坐标系下的描述
    R_c2p = np.array(
        [[0.99970737, - 0.00727533, 0.02307026],
         [0.01073077, 0.98810691, - 0.15339354],
         [-0.0216799,
          0.15359621,
          0.98789584]]
    )
    t_c2p = np.array((
        [[-68.82968479],
         [0.40231569],
         [7.98265215]]))

    # 重力向量，两个向量完全一致时遇到过拟合平面的SVD报错，至今未发现原因
    # 在实际使用中应该不会触发这个bug
    gravity_raw_standard = (2.5, 7, 0.45)  # 平放时的重力向量
    gravity_raw_using = (0.01, 9.8, 0.01)  # 此刻的重力向量
    # 画面和照片
    img_p = cv2.imread("../image_when_use/up.jpg")
    img_c = cv2.imread("../image_when_use/uc7.jpg")

    # 实例化一个矫正器
    correction = KeystoneCorrection(camera_mtx, camera_dist,
                                    projector_mtx, projector_dist, projector_resolution,
                                    R_c2p, t_c2p, gravity_raw_standard,
                                    gravity_raw_using, img_c, img_p, False)
    # 跟矫正器要offsets，欧拉角
    offsets1 = correction.offsets
    pose1 = correction.pose_wall_in_p
    get_adb_command(offsets1)

    # 如果换了一个姿势，想再做一次新的矫正，可以这样用
    img_c = cv2.imread("../image_when_use/uc7.jpg")  # 捕获一个新的相机照片
    gravity_raw_using = (0.01, 9.8, 0.01)  # 新的重力向量
    correction.update(img_c, gravity_raw_using)
    offsets2 = correction.offsets
    pose2 = correction.pose_wall_in_p
    get_adb_command(offsets2)
