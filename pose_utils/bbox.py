import cv2
import colorsys
import logging
import numpy as np
import time
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from copy import copy

# logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

class BBoxEstimation:
    def __init__(self, device=None) -> None:
        # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        # num_classes = 2
        # in_features = model.roi_heads.box_predictor.cls_score.in_features
        # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        self.model = model
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
    def get_bbox(self, img_bgr, vis=False) -> list:
        mask_model = self.model
        device = self.device
        
        width, height, _ = img_bgr.shape
        mask_model = mask_model.to(device)
        
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        transformer = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        
        img_torch = transformer(img)
        img_torch = torch.unsqueeze(img_torch, 0)
        img_torch = img_torch.to(self.device)
        
        mask_model.eval()
        
        start = time.time()
        with torch.no_grad():
            output = mask_model(img_torch)
        logger.info('estimated time:{}'.format(time.time() - start))
        data = output[0]
        
        scores = data['scores'].cpu().detach().numpy().copy()
        boxes = data['boxes'].cpu().detach().numpy().copy()
        labels = data['labels'].cpu().detach().numpy().copy()
        
        valid = labels == 1
        valid = valid & (scores > 0.5)
        boxes = boxes[valid]
        logger.debug(scores[valid])
        
        if vis:
            boxes_int = boxes.astype(np.int32)
            img_show = img_bgr.copy()
            num_person = len(boxes_int)
            for i, box in enumerate(boxes_int):
                col = i / num_person , 1.0, 1.0
                col = colorsys.hsv_to_rgb(*col)
                col = tuple(c * 255 for c in col)
                logger.debug(col)
                img_show = cv2.rectangle(img_show, box[:2], box[2:], col, thickness=5, lineType=cv2.LINE_AA)
            cv2.imwrite('bbox.png', img_show)
            # cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        # Change bbox [x1, y1, x2, y2] -> [x, y, w, h]
        boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
        boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
        valid2 = (boxes[:, 2] > (width * .1)) | (boxes[:, 3] > (height * .1))
        boxes = boxes[valid2]
        logger.debug(boxes)
        return boxes.tolist()
