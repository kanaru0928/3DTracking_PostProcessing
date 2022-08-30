import cv2
import socket
import numpy as np

from pose_estimation import PoseEstimation
from pose_utils.utils import JointInfo

pose_factory = None
cap = cv2.VideoCapture(6)

def get_joint(file_path=None):
    if file_path is not None:
        img = cv2.imread(file_path)
    else:
        _, img = cap.read()
    angles, joint3d, root_rot = pose_factory.estimation(img)
    if len(angles) == 0: return JointInfo()
    # print('shape:{}'.format(joint3d.shape))
    root_pos = joint3d[:, 14, :]
    # print('sliced:{}'.format(root_pos))
    return JointInfo(angles, root_pos, root_rot)

def loop(soc: socket.socket):
    joint = get_joint()
    soc.send('101'.encode('utf-8'))
    # print('sand {}'.format('101'.encode('utf-8')))
    # print('sending:{}'.format(joint))
    mes = joint.to_bytes()
    assert soc.recv(1024) == b'101'
    
    soc.send(mes)

def send_joint():
    host = 'localhost'
    port = 8111
    soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    soc.connect((host, port))
    try:
        while True:
            loop(soc)
    finally:
        soc.close()

def main():
    send_joint()

if __name__ == "__main__":
    pose_factory = PoseEstimation()
    main()
