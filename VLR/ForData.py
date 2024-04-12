import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from PIL import Image
import json
import cv2
import math
from tqdm import tqdm


class YOLOv8_face:
    def __init__(self, path, conf_thres=0.2, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        self.class_names = ['face']
        self.num_classes = len(self.class_names)
        
        # Initialize model
        self.net = cv2.dnn.readNet(path)
        self.input_height = 640
        self.input_width = 640
        self.reg_max = 16

        self.project = np.arange(self.reg_max)
        self.strides = (8, 16, 32)
        self.feats_hw = [(math.ceil(self.input_height / self.strides[i]), math.ceil(self.input_width / self.strides[i])) for i in range(len(self.strides))]
        self.anchors = self.make_anchors(self.feats_hw)

    def make_anchors(self, feats_hw, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points = {}
        for i, stride in enumerate(self.strides):
            h,w = feats_hw[i]
            x = np.arange(0, w) + grid_cell_offset  # shift x
            y = np.arange(0, h) + grid_cell_offset  # shift y
            sx, sy = np.meshgrid(x, y)
            # sy, sx = np.meshgrid(y, x)
            anchor_points[stride] = np.stack((sx, sy), axis=-1).reshape(-1, 2)
        return anchor_points

    def softmax(self, x, axis=1):
        x_exp = np.exp(x)
        # 如果是列向量，则axis=0
        x_sum = np.sum(x_exp, axis=axis, keepdims=True)
        s = x_exp / x_sum
        return s
    
    def resize_image(self, srcimg, keep_ratio=True):
        top, left, newh, neww = 0, 0, self.input_width, self.input_height
        if keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.input_height, int(self.input_width / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.input_width - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.input_width - neww - left, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))  # add border
            else:
                newh, neww = int(self.input_height * hw_scale), self.input_width
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.input_height - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.input_height - newh - top, 0, 0, cv2.BORDER_CONSTANT,
                                         value=(0, 0, 0))
        else:
            img = cv2.resize(srcimg, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, srcimg):
        input_img, newh, neww, padh, padw = self.resize_image(cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB))
        scale_h, scale_w = srcimg.shape[0]/newh, srcimg.shape[1]/neww
        input_img = input_img.astype(np.float32) / 255.0

        blob = cv2.dnn.blobFromImage(input_img)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        # if isinstance(outputs, tuple):
        #     outputs = list(outputs)
        # if float(cv2.__version__[:3])>=4.7:
        #     outputs = [outputs[2], outputs[0], outputs[1]] ###opencv4.7需要这一步，opencv4.5不需要
        # Perform inference on the image
        det_bboxes, det_conf, det_classid, landmarks = self.post_process(outputs, scale_h, scale_w, padh, padw)
        return det_bboxes, det_conf, det_classid, landmarks

    def post_process(self, preds, scale_h, scale_w, padh, padw):
        bboxes, scores, landmarks = [], [], []
        for i, pred in enumerate(preds):
            stride = int(self.input_height/pred.shape[2])
            pred = pred.transpose((0, 2, 3, 1))
            
            box = pred[..., :self.reg_max * 4]
            cls = 1 / (1 + np.exp(-pred[..., self.reg_max * 4:-15])).reshape((-1,1))
            kpts = pred[..., -15:].reshape((-1,15)) ### x1,y1,score1, ..., x5,y5,score5

            # tmp = box.reshape(self.feats_hw[i][0], self.feats_hw[i][1], 4, self.reg_max)
            tmp = box.reshape(-1, 4, self.reg_max)
            bbox_pred = self.softmax(tmp, axis=-1)
            bbox_pred = np.dot(bbox_pred, self.project).reshape((-1,4))

            bbox = self.distance2bbox(self.anchors[stride], bbox_pred, max_shape=(self.input_height, self.input_width)) * stride
            kpts[:, 0::3] = (kpts[:, 0::3] * 2.0 + (self.anchors[stride][:, 0].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 1::3] = (kpts[:, 1::3] * 2.0 + (self.anchors[stride][:, 1].reshape((-1,1)) - 0.5)) * stride
            kpts[:, 2::3] = 1 / (1+np.exp(-kpts[:, 2::3]))

            bbox -= np.array([[padw, padh, padw, padh]])  ###合理使用广播法则
            bbox *= np.array([[scale_w, scale_h, scale_w, scale_h]])
            kpts -= np.tile(np.array([padw, padh, 0]), 5).reshape((1,15))
            kpts *= np.tile(np.array([scale_w, scale_h, 1]), 5).reshape((1,15))

            bboxes.append(bbox)
            scores.append(cls)
            landmarks.append(kpts)

        bboxes = np.concatenate(bboxes, axis=0)
        scores = np.concatenate(scores, axis=0)
        landmarks = np.concatenate(landmarks, axis=0)
    
        bboxes_wh = bboxes.copy()
        bboxes_wh[:, 2:4] = bboxes[:, 2:4] - bboxes[:, 0:2]  ####xywh
        classIds = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)  ####max_class_confidence
        
        mask = confidences>self.conf_threshold
        bboxes_wh = bboxes_wh[mask]  ###合理使用广播法则
        if bboxes_wh.size==0:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])
        confidences = confidences[mask]
        classIds = classIds[mask]
        landmarks = landmarks[mask]
        
        indices = cv2.dnn.NMSBoxes(bboxes_wh.tolist(), confidences.tolist(), self.conf_threshold,
                                   self.iou_threshold).flatten()
        if len(indices) > 0:
            mlvl_bboxes = bboxes_wh[indices]
            confidences = confidences[indices]
            classIds = classIds[indices]
            landmarks = landmarks[indices]
            return mlvl_bboxes, confidences, classIds, landmarks
        else:
            print('nothing detect')
            return np.array([]), np.array([]), np.array([]), np.array([])

    def distance2bbox(self, points, distance, max_shape=None):
        x1 = points[:, 0] - distance[:, 0]
        y1 = points[:, 1] - distance[:, 1]
        x2 = points[:, 0] + distance[:, 2]
        y2 = points[:, 1] + distance[:, 3]
        if max_shape is not None:
            x1 = np.clip(x1, 0, max_shape[1])
            y1 = np.clip(y1, 0, max_shape[0])
            x2 = np.clip(x2, 0, max_shape[1])
            y2 = np.clip(y2, 0, max_shape[0])
        return np.stack([x1, y1, x2, y2], axis=-1)
    
    def draw_detections(self, image, boxes, scores, kpts):
        for box, score, kp in zip(boxes, scores, kpts):
            x, y, w, h = box.astype(int)
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), thickness=1)
            cv2.putText(image, "face:"+str(round(score,2)), (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness=1)
            for i in range(5):
                cv2.circle(image, (int(kp[i * 3]), int(kp[i * 3 + 1])), 2, (0, 255, 0), thickness=-1)
                # cv2.putText(image, str(i), (int(kp[i * 3]), int(kp[i * 3 + 1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), thickness=1)
        return image

modelpath = '/ocean/projects/cis220031p/abdulhan/VLR/yolov8n-face_demo.onnx'
confThreshold = 0.45
nmsThreshold = 0.45 

# Initialize YOLOv8_face object detector
YOLOv8_face_detector = YOLOv8_face(modelpath, conf_thres=confThreshold, iou_thres=nmsThreshold)

from PIL import Image
def scale_img(boxes, scale, img_path, mask_path,  new_img_path, new_mask_path):
    for i in range(len(boxes)):
        original_image = Image.open(img_path)
        original_mask = Image.open(mask_path)
        # top_left = (boxes[0][0], boxes[0][1]) 
        max_dim = max(boxes[0][2], boxes[0][3])

        side_length = max_dim * scale
        center_x = boxes[i][0] + (boxes[i][2]/2)
        center_y = boxes[i][1] + (boxes[i][3]/2)
        half_side_length = side_length // 2
        # Get the width and height of the image
        image_width, image_height = original_image.size

        # Calculate bounding box corners
        x1 = max(0, center_x - half_side_length)
        y1 = max(0, center_y - half_side_length)
        x2 = min(image_width, center_x + half_side_length)
        y2 = min(image_height, center_y + half_side_length)

        # Adjust width or height to make the box square
        box_width = x2 - x1
        box_height = y2 - y1

        if box_width > box_height:
            # Expand height
            diff = box_width - box_height
            y1 = max(0, y1 - diff // 2)
            y2 = min(image_height, y2 + diff // 2)
        elif box_height > box_width:
            # Expand width
            diff = box_height - box_width
            x1 = max(0, x1 - diff // 2)
            x2 = min(image_width, x2 + diff // 2)


        top_left = (x1, y1)
        bottom_right = (x2, y2)
        # bottom_right = ((boxes[0][0] + max_dim) * scale, (boxes[0][1] + max_dim) * scale)
        cropped_image = original_image.crop((*top_left, *bottom_right))
        cropped_image.save(new_img_path + '_cropped_' + str(i) + '.jpg')
        cropped_mask = original_mask.crop((*top_left, *bottom_right))
        cropped_mask.save(new_mask_path + '_cropped_' + str(i) + '.png')
        print("New Image Saved")


img_path = '/ocean/projects/cis220031p/abdulhan/VLR/P3M-10k/train/blurred_image'
mask_path = '/ocean/projects/cis220031p/abdulhan/VLR/P3M-10k/train/mask'

img_file_names = sorted([f for f in os.listdir(img_path)])
mask_file_names = sorted([f for f in os.listdir(mask_path)])

img_files = [os.path.join(img_path, file_name) for file_name in img_file_names]
mask_files = [os.path.join(mask_path, file_name) for file_name in mask_file_names]


new_img_file_path = '/ocean/projects/cis220031p/abdulhan/VLR/testVLR/images/'
new_mask_file_path = '/ocean/projects/cis220031p/abdulhan/VLR/testVLR/masks/'
for img_path, mask_path, img_file_name, mask_file_name in zip(img_files, mask_files, img_file_names, mask_file_names):
    img_path = img_path
    mask_path = mask_path
    img = cv2.imread(img_path)
    boxes, scores, classids, kpts = YOLOv8_face_detector.detect(img)
    print(boxes)
    scale_img(boxes=boxes, scale=2, img_path=img_path, mask_path=mask_path, new_img_path=(new_img_file_path+img_file_name[:-4]), new_mask_path=(new_mask_file_path+mask_file_name[:-4]))
