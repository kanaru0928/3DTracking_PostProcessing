import argparse
import cv2
import logging
import math
import numpy as np
from pprint import pprint
from queue import Queue
import sys, os
import socket
import time
import torch

import rootnet.demo.demo as rootnet
import posenet.demo.demo as posenet

from pose_utils.config import Config

if 'yolov5' in Config.detectNet:
    from pose_utils.bbox_yolo import BBoxEstimation
else:
    from pose_utils.bbox import BBoxEstimation

logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

def rotate_z(vec:np.ndarray, theta:float) -> np.ndarray:
    S = math.sin(theta)
    C = math.cos(theta)
    rotate_matrix = np.array([
        [C, -S, 0],
        [S,  C, 0],
        [0,  0, 1]
    ])
    ret = np.dot(rotate_matrix, vec)
    return ret
    
def rotate_y(vec:np.ndarray, theta:float) -> np.ndarray:
    S = math.sin(theta)
    C = math.cos(theta)
    rotate_matrix = np.array([
        [ C, 0, S],
        [ 0, 1, 0],
        [-S, 0, C]
    ])
    ret = np.dot(rotate_matrix, vec)
    return ret

class PoseEstimation():
    def __init__(self, device=None) -> None:
        self.num_joint = 21
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        # self.img = self.img.to()
        self.root_joint = 14
        self.root = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1, 4, 7, 10, 13]
        self.tree = [[] for _ in range(self.num_joint)]
        for i, j in enumerate(self.root):
            if j == -1: continue
            self.tree[j].append(i)
        
        self.bbox_factory = BBoxEstimation(device=self.device)
        self.root_factory = rootnet.Rootnet(rootnet.Args())
        self.pose_factory = posenet.Posenet(posenet.Args())
    
    @staticmethod
    def conv_to_euler(rel_pos : np.ndarray):
        ret = np.zeros(3)
        
        # # With rotate matrix
        # ret[1] = math.atan2(rel_pos[2], rel_pos[0])
        # rel_pos = rotate_y(rel_pos, -ret[1])
        # ret[2] = math.atan2(rel_pos[1], rel_pos[0])
        
        # With dot production
        # xy_pos = rel_pos.copy()
        # xy_pos[2] = 0
        # pos_size = np.linalg.norm(rel_pos)
        # xy_pos_size = np.linalg.norm(xy_pos)
        # if rel_pos[2] < 0:
        #     ret[1] = math.acos(xy_pos_size / pos_size)
        # else:
        #     ret[1] = -math.acos(xy_pos_size / pos_size)
        # if rel_pos[1] > 0:
        #     ret[2] = math.acos(rel_pos[0] / xy_pos_size)
        # else:
        #     ret[2] = -math.acos(rel_pos[0] / xy_pos_size)
        
        pos = rel_pos.copy()
        size = np.linalg.norm(pos)
        pos = pos / size
        
        ret[1] = -math.asin(pos[2])
        ret[2] = np.sign(pos[0]) * math.atan(pos[1] / pos[2])
        
        return ret
    
    def gen_angle(self, joints):
        num_joint = self.num_joint
        num_person = len(joints)
        # angle = np.zeros((num_person, num_joint, 3))
        rel_vec = np.zeros((num_person, num_joint, 3))
        rel_angle = np.zeros((num_person, num_joint, 3))
        
        root_rot = []

        for k, joint in enumerate(joints):
            # Calcurate root rotation
            rel_lhip = joint[11] - joint[self.root[11]]
            
            lhip_y = self.conv_to_euler(rel_lhip)[1]
            
            # lhip_z = math.atan2(rel_lhip[1], rel_lhip[0])
            # rel_lhip = rotate_z(rel_lhip, -lhip_z)
            # lhip_y = math.atan2(rel_lhip[0], rel_lhip[2])
            
            root_rot.append(lhip_y)
            
            # Calcurate absolution angle using BFS
            que = Queue()
            que.put(self.root_joint)
            while not que.empty():
                idx = que.get()
                if idx != self.root_joint:
                    rel_pos = joint[idx] - joint[self.root[idx]]
                    # rel_pos = rotate_y(rel_pos, lhip_y)

                    # angle[k][idx] = self.conv_to_euler(rel_pos)
                    rel_vec[k][idx] = rel_pos
                    
                    # diff = angle[k][idx] - angle[k][self.root[idx]]
                    # rel_angle[k][idx] = diff
                
                for j in self.tree[idx]:
                    que.put(j)
            # abs_list = [2, 5, 9, 12, 15]
            # rel_angle[k][abs_list] = angle[k][abs_list]
        return rel_vec, np.array(root_rot)
    
    def estimation(self, img, vis1 = False, vis2 = False, vis3 = False):
        start_all = time.time()
        start = time.time()
        bbox_list = self.bbox_factory.get_bbox(img, vis1)
        logger.info('bbox:{}s'.format(time.time() - start))
        start = time.time()
        root_depth_list = self.root_factory.get_root(bbox_list, img, vis2)
        logger.info('root:{}s'.format(time.time() - start))
        start = time.time()
        joint3d = self.pose_factory.get_pose(img, bbox_list, root_depth_list, vis3)
        logger.info('pose:{}s'.format(time.time() - start))
        
        if len(joint3d) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # joint3d[:,:,0] = -joint3d[:,:,0]
        joint3d[:,:,1] = -joint3d[:,:,1]
        # joint3d[:,:,2] = -joint3d[:,:,2]
        
        angles, root_rot = self.gen_angle(joint3d)
        # logger.debug(joint3d.tolist())
        # logger.debug(angles.tolist())
        
        logger.info('alltime:{}s'.format(time.time() - start_all))
        
        return joint3d, joint3d, root_rot

def init():
    def exec():
        nonlocal pose_factory
        img = cv2.imread(os.path.join(os.path.dirname(__file__), 'media/input5.jpg'))
        assert img is not None
        angles, joints = pose_factory.estimation(img)
        ret_str = angles.astype('str')
        ret_str = [[' '.join(y) for y in x] for x in ret_str]
        ret_str = [','.join(x) for x in ret_str]
        ret_str = ';'.join(ret_str)
        pos_str = [' '.join(x) for x in joints[:,14,:]]
        pos_str = ';'.join(pos_str)
        
        return ret_str + '/' + pos_str
    pose_factory = PoseEstimation()
    return exec

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default=None, dest='img_path')
    return parser.parse_args()

def console():
    args = parse()
    img_path = args.img_path
    # img = cv2.imread(os.path.join(os.path.dirname(__file__), 'media/input6.jpg'))
    cap = cv2.VideoCapture(6)
    _, img = cap.read()
    assert img is not None
    pose_factory = PoseEstimation('cpu')
    angles, joints, root_rot = pose_factory.estimation(img, vis1=True, vis2=True, vis3=True)
    ret_str = angles.astype('str')
    joints = joints.astype('str')
    root_rot = np.array(root_rot, dtype='str')
    ret_str = [[' '.join(y) for y in x] for x in ret_str]
    ret_str = [','.join(x) for x in ret_str]
    ret_str = ';'.join(ret_str)
    pos_str = [' '.join(x) for x in joints[:,14,:]]
    pos_str = ';'.join(pos_str)
    root_str = ';'.join(root_rot)
    print(ret_str + '/' + pos_str + '/' + root_str)

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('TkAgg')
    console()