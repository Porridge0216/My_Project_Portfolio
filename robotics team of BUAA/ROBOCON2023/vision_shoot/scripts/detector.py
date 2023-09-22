#!/bin/python3
# -*- coding utf-8 -*-
import torch
from models.common import DetectMultiBackend
from utils.dataloaders import LoadStreams
from utils.general import (Profile, check_img_size, cv2,
                           non_max_suppression, scale_boxes, )
from utils.torch_utils import select_device, smart_inference_mode
from vision_shoot.msg import pole as onepole, poles
import rospy


class YoloDetector:
    def __init__(self):
        # ROS初始化
        rospy.init_node("vision_detector_node", anonymous=True)  # 创建节点
        self.vision_pub = rospy.Publisher(
            "detect_result", poles, queue_size=1)  # 发布识别到的柱子的角度

        self.poles = poles()
        self.poles.data_list = []
        self.rate = rospy.Rate(30)
    
    @smart_inference_mode()
    def run(self,
            weights='/home/br/supporting_ws/best.pt',  # model path or triton URL
            data='/home/br/supporting_ws/data.yaml',  # dataset.yaml path
            conf_thres=0.7,  # confidence threshold
            iou_thres=0.45,  # NMS IOU threshold
            ):

        # count = 0
        # Load model
        device = select_device('cpu')
        model = DetectMultiBackend(weights, device=device, dnn=False, data=data, fp16=False)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size((640, 640), s=stride)  # check image size

        # Dataloader
        dataset = LoadStreams('0', img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
        bs = len(dataset)  # batch_size
        
        # Run inference
        model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
        
        for path, im, im0s, vid_cap, s in dataset:
            with dt[0]:
                im = torch.from_numpy(im).to(model.device)
                im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                im /= 255  # 0 - 255 to 0.0 - 1.0
                if len(im.shape) == 3:
                    im = im[None]  # expand for batch dim

            # Inference
            with dt[1]:
                pred = model(im)
                

            # NMS
            with dt[2]:
                pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)
                
            # Second-stage classifier (optional)
            # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

            # Process predictions
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0 = im0s[i].copy()
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                #print(det)
                #print("------------")
                # Stream results

                #cv2.namedWindow('res', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                #cv2.resizeWindow('res', int(im0.shape[1]/2), int(im0.shape[0]/2))
                
                
                self.poles = poles()
                for d in det:
                    *xyxy, conf, class_index = d
                    new_pole_msg = onepole()
                    new_pole_msg.x = ((xyxy[0] + xyxy[2]) / 2)
                    new_pole_msg.y = ((xyxy[1] + xyxy[3]) / 2)
                    new_pole_msg.w = xyxy[2] - xyxy[0]
                    new_pole_msg.h = xyxy[3] - xyxy[1]

                    self.poles.data_list.append(new_pole_msg)

                    pt1 = (int(xyxy[0]), int(xyxy[1]))
                    pt2 = (int(xyxy[2]), int(xyxy[3]))
                    # cv2.rectangle(im0, pt1, pt2, (0, 0, 255), 3)
                self.vision_pub.publish(self.poles)
                
                
                im0 = cv2.resize(im0,(960,540)) 
                # cv2.imshow('res', im0)
                # cv2.waitKey(1)  # 1 millisecond

    def main(self):
        self.run()


if __name__ == "__main__":
    detector = YoloDetector()
    detector.main()
    rospy.spin()
