#!/bin/python3
# -*- coding utf-8 -*-
import copy
import tf
import tf.transformations as tftr
from vision_shoot.srv import shoot_aid, shoot_aidResponse
from vision_shoot.msg import head_angle
import numpy as np
from math import atan, sin, cos, sqrt, asin
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (Profile, check_img_size, cv2,
                           non_max_suppression, scale_boxes, )
from utils.torch_utils import select_device, smart_inference_mode
import rospy

PI = 3.14159265
VISIBLE_ANGLE_THRESHOLD = 5
MATCH_ERROR_THRESHOLD = 5
ANGLE_BIAS_ERROR_THRESHOLD = 10
DISTANCE_THRESHOLD = 2000
"""
所有角度都是 (-180,180]
"""


class Pole:
    def __init__(self, x=None, y=None, index=None, top2ground=None):
        self.x = x
        self.y = y
        self.top2ground = top2ground  # 顶部到一层地的高度

        self.angle = None
        self.distance = None
        self.pixcel_y = None
        self.pixcel_x = None
        self.pixcel_w = None
        self.pixcel_h = None

        self.id = index

    def to(self, x2map, y2map, angle2map):
        angle = 0
        try:
            # 斜率
            k = (self.y - y2map) / (self.x - x2map)
            angle = atan(k) / PI * 180
            if k > 0:
                if self.y < y2map:
                    angle -= 180
            elif k < 0:
                if self.y > y2map:
                    angle += 180
            else:
                if self.x < x2map:
                    angle += 180

        except ZeroDivisionError:
            if self.y < y2map:
                angle = -90
            elif self.y > y2map:
                angle = 90

        angle = angle - angle2map
        if angle <= -180:
            angle += 360

        distance = sqrt((self.x - x2map) ** 2 + (self.y - y2map) ** 2)
        return angle, distance


class Camera:
    def __init__(self):
        # 内参
        self.intrinsic_mtx = np.zeros(9, dtype='float32')
        self.distortion_coefficient = np.zeros(5, dtype='float32')
        self.cx = None
        self.cy = None
        self.fx = None
        self.fy = None
        self.visable_angle = None
        # 外参
        self.transform = np.zeros(2, dtype='float32')  # 相机对云台 向右为x 向前为y
        self.tx = None
        self.ty = None
        self.angle = rospy.get_param("camera_param/transform/angle", 0)  # 相对云台的角度（误差）逆时针为正
        self.height = rospy.get_param('camera_param/height', 1)  # 对地高度

        # 分辨率
        self.resolution = np.zeros(2, dtype='float32')
        self.frame_height = None
        self.frame_width = None

        for i in range(9):
            self.intrinsic_mtx[i] = rospy.get_param('camera_param/mtx/mtx' + str(i), 1)
        self.intrinsic_mtx = self.intrinsic_mtx.reshape((3, 3))

        for i in range(5):
            self.distortion_coefficient[i] = rospy.get_param('camera_param/dist/dist' + str(i), 0)
        self.distortion_coefficient = self.distortion_coefficient.reshape((1, 5))

        for i in range(2):
            self.resolution[i] = rospy.get_param('camera_param/resolution/resolution' + str(i), 1)
        self.resolution = self.resolution.reshape((1, 2))

        for i in range(2):
            self.transform[i] = rospy.get_param("camera_param/transform/transform" + str(i), 0)
        self.transform = self.transform.reshape((1, 2))

        self.fx = self.intrinsic_mtx[0][0]
        self.fy = self.intrinsic_mtx[1][1]
        self.cx = self.intrinsic_mtx[0][2]
        self.cy = self.intrinsic_mtx[1][2]

        self.tx = self.transform[0][0]
        self.ty = self.transform[0][1]

        self.frame_width = self.resolution[0][0]
        self.frame_height = self.resolution[0][1]

        self.visable_angle = atan((self.frame_width / 2) / self.fx) / PI * 180 * 2

    def to_map(self, head_xy2map, head_angle2map):
        """
        Args:
            head_xy2map: 元组，云台到地图的xy
            head_angle2map: float，云台到地图的旋转角 逆时针为正 x轴为0
        Returns: 相机到地图的xy，角度
        """
        x = head_xy2map[0] + self.tx * sin(head_angle2map / 180 * PI) + self.ty * cos(head_angle2map / 180 * PI)
        y = head_xy2map[1] + self.ty * sin(head_angle2map / 180 * PI) - self.tx * cos(head_angle2map / 180 * PI)
        angle = head_angle2map + self.angle
        if angle > 180:
            angle -= 360
        return (x, y), angle

    def print_info(self):
        print("内参：")
        print(self.intrinsic_mtx)
        print("cx:", self.cx)
        print("cy:", self.cy)
        print("fx", self.fx)
        print("fy", self.fy)
        print("visible_angle", self.visable_angle)

        print("到云台xy angle：")
        print(self.transform)
        print("tx:", self.tx)
        print("ty:", self.ty)
        print("angle:", self.angle)

        print("高度")
        print(self.height)

        print("分辨率：")
        print(self.resolution)
        print("width:", self.frame_width)
        print("height:", self.frame_height)


class Head:
    def __init__(self):
        self.angle = 0  # 相对车，偏左为正 逆时针为正 朝前为0
        self.transform = np.zeros(2, dtype='float32')  # 相对底盘 向右为x，向前为y
        self.tx = None
        self.ty = None
        for i in range(2):
            self.transform[i] = rospy.get_param("head_param/transform/transform" + str(i), 1)

        self.transform = self.transform.reshape((1, 2))
        self.tx = self.transform[0][0]
        self.ty = self.transform[0][1]

    def to_map(self, robot_xy2map, robot_angle2map):
        """
        Args:
            robot_xy2map: 元组，机器人到地图的xy
            robot_angle2map: float，机器人到地图的旋转角
        Returns: 云台到地图的xy，角度
        """
        x = robot_xy2map[0] + self.tx * sin(robot_angle2map / 180 * PI) + self.ty * cos(robot_angle2map / 180 * PI)
        y = robot_xy2map[1] + self.ty * sin(robot_angle2map / 180 * PI) - self.tx * cos(robot_angle2map / 180 * PI)
        angle = robot_angle2map + self.angle

        if angle > 180:
            angle -= 360

        return (x, y), angle

    def print_info(self):
        print("到底盘xy：")
        print(self.transform)
        print("tx:", self.tx)
        print("ty:", self.ty)


class ShootNode:
    @smart_inference_mode()
    def __init__(self, weight):
        # 相机初始化
        print("======初始化Camera======")
        self.camera = Camera()
        self.ifshow = rospy.get_param("show", 1)
        self.camera.print_info()
        print("ifshow:", self.ifshow)

        # Yolo初始化
        print("======初始化Yolo======")
        data = '/home/br/supporting_ws/src/vision_shoot/data.yaml',  # dataset.yaml path

        # Load model
        device = select_device('cpu')
        self.model = DetectMultiBackend(weight, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = self.model.stride, self.model.names, self.model.pt
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        # Dataloader
        try:
            self.dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        except:
            try:
                self.dataset = LoadStreams('1', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
            except:
                self.dataset = LoadStreams('2', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

        bs = len(self.dataset)  # batch_size
        self.dt = (Profile(), Profile(), Profile())
        # Run inference
        self.model.warmup(imgsz=(1 if pt or self.model.triton else bs, 3, *imgsz))  # warmup

        # 云台初始化
        print("======初始化云台======")
        self.head = Head()
        self.head.print_info()

        # 变量
        self.counter = 0
        self.image = None
        self.show_image = None
        self.target_id = None  # 要射击的柱子id
        self.target_detected_xywh = []  # 识别到的所有柱子的xywh
        self.aid_state = True  # 帮助状态
        self.poles_detected = []  # 检测到的柱子
        self.poles_matched = []  # 匹配成功的柱子
        self.poles_pri2cam = []  # 先验证下的柱子到相机

        self.camera_x2map_pri = None
        self.camera_y2map_pri = None
        self.camera_angle2map_pri = None

        self.head_x2map_pri = None
        self.head_y2map_pri = None
        self.head_angle2map_pri = None

        self.target_exist = False
        self.target_matched = None

        self.angle_bias = 0  # 要发布的角度偏移量

        pole0 = Pole(2750, 2750, 0, 1000)
        pole1 = Pole(6000, 2750, 1, 1000)
        pole2 = Pole(9250, 2750, 2, 1000)

        pole3 = Pole(4650, 4650, 3, 1200)
        pole4 = Pole(7350, 4650, 4, 1200)

        pole5 = Pole(4650, 7350, 5, 1200)
        pole6 = Pole(7350, 7350, 6, 1200)

        pole7 = Pole(2750, 9250, 7, 1000)
        pole8 = Pole(6000, 9250, 8, 1000)
        pole9 = Pole(9250, 9250, 9, 1000)

        pole10 = Pole(6000, 6000, 10, 1900)

        self.pole_list = [pole0, pole1, pole2, pole3, pole4, pole5, pole6, pole7, pole8, pole9, pole10]


        # ROS初始化
        print("======初始化ROS======")
        rospy.init_node("shoot", anonymous=True)  # 创建节点
        rospy.Subscriber('/head_angle', head_angle, self.head_angle_callback)  # 订阅云台角度
        rospy.Service("/shoot_aid", shoot_aid, self.aid)  # 射击辅助服务
        self.transformer_listener = tf.TransformListener()  # tf监听器
        rospy.loginfo("节点初始化完成，等待服务调用")
        # cv2.namedWindow('res', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        # cv2.resizeWindow('res', 960, 540)

    @staticmethod
    def main():
        while not rospy.is_shutdown():
            rospy.spin()

    def head_angle_callback(self, msg):
        """
        更新云台角度的回调函数
        """
        #self.counter +=1
        self.head.angle = msg.angle - 180
        
        #print("received head angle: ", self.head.angle)
        #print(self.counter)

    @smart_inference_mode()
    def detect(self,
               conf_thres=0.7,  # confidence threshold
               iou_thres=0.45,  # NMS IOU threshold
               ):
        for path, im, im0s, vid_cap, s in self.dataset:
            with self.dt[0]:
                im = torch.from_numpy(im).to(self.model.device)
                im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with self.dt[1]:
                pred = self.model(im)

            # NMS
            with self.dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                im0 = im0s[i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                target_detected_xywh = []
                for d in det:
                    *xyxy, conf, class_index = d
                    p = [0, 0, 0, 0]
                    p[0] = ((xyxy[0] + xyxy[2]) / 2).item()
                    p[1] = ((xyxy[1] + xyxy[3]) / 2).item()
                    p[2] = abs((xyxy[2] - xyxy[0]).item())
                    p[3] = abs((xyxy[3] - xyxy[1]).item())

                    target_detected_xywh.append(p)

                    pt1 = (int(xyxy[0]), int(xyxy[1]))
                    pt2 = (int(xyxy[2]), int(xyxy[3]))
                    cv2.rectangle(im0, pt1, pt2, (255, 255, 0), 3)
                im0 = cv2.resize(im0, (960, 540))
                self.target_detected_xywh = target_detected_xywh
                self.image = im0
            break

    def aid(self, req):
        """
        响应客户端请求的回调函数
        """
        print("======开始调用射击辅助======")
        # 更新要打的柱子id
        self.target_id = req.target_id
        print("processing request id : ", self.target_id)
        self.target_exist = False
        # 抓取所有识别到的柱子
        self.detect()
        self.detect_process()  # 会同时更新self.aid_state, 如果识别结果大于等于1个，为true，如果图像中断或者没有识别出来，为false
        print("detected poles count : ", len(self.poles_detected))
        print("detect result:")
        for p in self.poles_detected:
            print(p.angle)

        # 更新机器人位姿
        tf  = True
        try:
            robot_xyz2map, robot_orint = self.transformer_listener.lookupTransform("/map", "/base_footprint", rospy.Time(0))
            robot_xy2map = (robot_xyz2map[0] * 1000, robot_xyz2map[1] * 1000)
            euler = tftr.euler_from_quaternion(robot_orint)
            robot_angle2map = euler[2] + 90
            print("From TF")
        except:
            robot_xy2map = (6000, 790-191.76)
            robot_angle2map = 90
            tf = False

        #robot_xy2map = (6000, 790-191.76)
        #robot_angle2map = 90

        if robot_angle2map > 180:
            robot_angle2map -= 360

        
        # 更新云台位姿先验
        (self.head_x2map_pri, self.head_y2map_pri), self.head_angle2map_pri = self.head.to_map(robot_xy2map, robot_angle2map)
       
        # 更新相机位姿先验
        (self.camera_x2map_pri, self.camera_y2map_pri), self.camera_angle2map_pri = self.camera.to_map((self.head_x2map_pri, self.head_y2map_pri),
                                                                                                       self.head_angle2map_pri)

        # 更新全场柱子先验 到相机
        self.poles_pri2cam = []
        print("柱子先验：")
        for p in self.pole_list:
            print("===")
            pole = copy.deepcopy(p)
            angle, distance = p.to(self.camera_x2map_pri, self.camera_y2map_pri, self.camera_angle2map_pri)

            if abs(angle) < self.camera.visable_angle / 2 + VISIBLE_ANGLE_THRESHOLD:
                pole.distance = distance
                pole.angle = angle
                self.poles_pri2cam.append(pole)
                print("id:", pole.id, ",", pole.angle)

        # 匹配
        if self.aid_state:
            self.match2()  # 会更新self.aid_state， 如果匹配失败，为false
            print("match_result：")
            for p in self.poles_matched:
                print("angle: ", p.angle, "  id: ", p.id if p.id is not None else "failed")

        # 计算
        self.calculate_bias()

        # 响应请求，返回云台bias
        print("======调用结束======")
        print("target_id:", self.target_id)
        print("车定位：", robot_xy2map, robot_angle2map)
        print("云台先验：", self.head_x2map_pri, ",", self.head_y2map_pri, self.head_angle2map_pri)
        print("相机先验：", self.camera_x2map_pri, ",", self.camera_y2map_pri, self.camera_angle2map_pri)

        print("angle_bias", self.angle_bias)
        s1 = "location: "+str(int(robot_xy2map[0]))+" , "+str(int(robot_xy2map[1]))+" , "+str(int(robot_angle2map))
        s2 = "head:    "+str(round(self.head.angle,2))
        s3 = "target:   "+str(self.target_id)
        s4 = "bias:     "+str(round(self.angle_bias,2))
        cv2.putText(self.show_image,s1 , (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 0), 3)
        cv2.putText(self.show_image,s2 , (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 200), 3)
        cv2.putText(self.show_image,s3 , (10,90), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 200, 0), 3)
        cv2.putText(self.show_image,s4 , (10,120), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 200), 3)
        cv2.line(self.show_image, (int(self.camera.cx / 2), 100), (int(self.camera.cx / 2), 440), (30, 30, 255), 2)
        cv2.line(self.show_image, (100, int(self.camera.cy / 2)), (860, int(self.camera.cy / 2)), (30, 30, 255), 2)


        cv2.imwrite("/home/br/supporting_ws/src/vision_shoot/scripts/res.png", self.show_image)
        if tf:
            rospy.logwarn("OK")
        else:
            rospy.logwarn("No tf !!!")


        return shoot_aidResponse(self.angle_bias)

    def match(self):
        for p in self.poles_pri2cam:
            print(p.angle)

        for p in self.poles_detected:
            wating_list = []
            for pp in self.poles_pri2cam:
                if abs(pp.angle - p.angle) < MATCH_ERROR_THRESHOLD:
                    wating_list.append(pp.id)

            if len(wating_list) == 1:
                p.id = wating_list[0]

        for p in self.poles_detected:
            for pp in self.poles_detected:
                if p is pp:
                    continue
                if p.id == pp.id:
                    p.id = None
                    pp.id = None

        self.poles_matched = []
        for p in self.poles_detected:
            if p.id is not None:
                self.poles_matched.append(p)
                if p.id == self.target_id:
                    self.target_exist = True
                    self.target_matched = p
                    break

        self.aid_state = (len(self.poles_matched) >= 1)

    def match2(self):
        self.poles_matched = []
        for p in self.poles_detected:
            print("=======")
            print("正在匹配柱子角度", p.angle)
            waiting_list = []
            for pp in self.poles_pri2cam:
                print(pp.id, "号柱子的角度先验是", pp.angle)
                if abs(pp.angle - p.angle) < MATCH_ERROR_THRESHOLD:
                    waiting_list.append(pp)

            print("waiting_list:")
            for ppp in waiting_list:
                print(ppp.id, " ", ppp.angle)

            if len(waiting_list) == 1:

                p.id = waiting_list[0].id
                p.distance = waiting_list[0].distance
                p.x = waiting_list[0].x
                p.y = waiting_list[0].y

                print("确定编号：", p.id)
            elif len(waiting_list) > 1:
                print("多个符合，看距离")
                wwaiting_list = []
                for pp in waiting_list:
                    print(pp.id, "号的先验距离：", pp.distance)
                    print("假设是：", pp.id)
                    print(self.camera.fy * abs(pp.top2ground - self.camera.height) / abs(p.pixcel_y - p.pixcel_h / 2 - self.camera.cy))

                    if abs(pp.distance - self.camera.fy * abs(pp.top2ground - self.camera.height) / abs(
                            p.pixcel_y - p.pixcel_h / 2 - self.camera.cy)) < DISTANCE_THRESHOLD:
                        wwaiting_list.append(pp)
                print("wwaiting_list:")
                for ppp in wwaiting_list:
                    print(ppp.id, " ", ppp.angle, " ", ppp.distance)

                if len(wwaiting_list) == 1:
                    p.id = wwaiting_list[0].id
                    p.distance = wwaiting_list[0].distance
                    p.x = wwaiting_list[0].x
                    p.y = wwaiting_list[0].y
                    print("确定编号：", p.id)

        for p in self.poles_detected:
            for pp in self.poles_detected:
                if p is pp:
                    continue
                if p.id == pp.id:
                    p.id = None
                    pp.id = None

        self.poles_matched = []
        for p in self.poles_detected:
            if p.id is not None:
                self.poles_matched.append(p)
                if p.id == self.target_id:
                    self.target_exist = True
                    print("exist")
                    self.target_matched = p

        for p in self.poles_matched:
            cv2.putText(self.show_image, str(p.id), (int(p.pixcel_x / 2), int(p.pixcel_y / 2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

        
        self.aid_state = (len(self.poles_matched) >= 1)
        

    def calculate_bias(self):
        if not self.aid_state:
            # 视觉失效 ， 直接通过定位和云台角度计算
            # 计算目标柱到云台的角度
            print("失效，用定位计算")
            self.angle_bias = self.pole_list[self.target_id].to(self.head_x2map_pri, self.head_y2map_pri, self.head_angle2map_pri)[0]
            return
        else:
            # 视觉辅助状态正常
            print("视觉辅助正常")
            angle_bias_pri = self.pole_list[self.target_id].to(self.head_x2map_pri, self.head_y2map_pri, self.head_angle2map_pri)[0]
            # 如果目标被直接匹配成功
            if self.target_exist:
                print("direct")
                self.angle_bias = atan((self.target_matched.distance * sin(self.target_matched.angle / 180 * PI)) / (
                        self.target_matched.distance * cos(self.target_matched.angle / 180 * PI) + self.camera.ty)) / PI * 180
                print("小角度修正后", self.angle_bias)
            else:
                sum_error = 0
                for p in self.poles_matched:
                    sum_error += (atan((p.distance * sin(p.angle / 180 * PI)) / (
                            p.distance * cos(p.angle / 180 * PI) + self.camera.ty)) / PI * 180) - (
                                     p.to(self.head_x2map_pri, self.head_y2map_pri, self.head_angle2map_pri)[0])

                mean_error = sum_error / len(self.poles_matched)
                self.angle_bias = angle_bias_pri + mean_error

            print("先验：", angle_bias_pri)
            print("后验：", self.angle_bias)
            if abs(self.angle_bias - angle_bias_pri) > ANGLE_BIAS_ERROR_THRESHOLD:
                print("误差过大，用定位计算")
                self.angle_bias = angle_bias_pri
            
            return

    def detect_process(self):
        """
        Returns:检测结果是否可供辅助
        """

        self.poles_detected = []
        # 更新self.poles_detected

        target_detected_xywh = self.target_detected_xywh
        self.show_image = self.image
        for p in target_detected_xywh:
            pole = Pole()
            pole.angle = -atan((p[0] - self.camera.cx) / self.camera.fx) / PI * 180
            pole.pixcel_x = p[0]
            pole.pixcel_y = p[1]
            pole.pixcel_w = p[2]
            pole.pixcel_h = p[3]

            self.poles_detected.append(pole)

        if len(self.poles_detected) >= 1:
            self.aid_state = True
            return
        else:
            self.aid_state = False
            return


if __name__ == "__main__":
    shoot_aid = ShootNode('/home/br/supporting_ws/src/vision_shoot/best.pt')
    shoot_aid.main()