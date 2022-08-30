import torch
import cv2
from torch import cudnn_affine_grid_generator
import torch.backends.cudnn as cudnn
from pose_utils.bbox_yolo import BBoxEstimation

def main():
    cudnn.fastest = True
    cudnn.benchmark

if __name__ == "__main__":
    main()
