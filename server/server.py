from ast import arg
import sys, os
import socket
import struct
import threading
import time
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pose_utils.utils import JointInfo, JointHandler

from pose_estimation import PoseEstimation
soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

host = 'localhost'
port = 8111

soc.bind((host, port))
soc.listen(5)

print('listening')

now_joint = bytes()
handler = JointHandler()

pose_factory = None
cap = cv2.VideoCapture(6)

send_continue = False
factory = PoseEstimation()

def get_joint(file_path=None):
    if file_path is not None:
        img = cv2.imread(file_path)
    else:
        _, img = cap.read()
    angles, joint3d, root_rot = factory.estimation(img)
    if len(angles) == 0: return JointInfo()
    # print('shape:{}'.format(joint3d.shape))
    root_pos = joint3d[:, 14, :]
    # print('sliced:{}'.format(root_pos))
    return JointInfo(angles, root_pos, root_rot)

def get_estimated_data(client : socket.socket):
    global now_joint
    res = client.recv(2048)
    now_joint = res
    # handler.set_joint(joint)
    print(now_joint)
    print('set joint')

def send_joint(client : socket.socket):
    # global now_joint
    joint = handler.get_joint()
    buf = joint.to_bytes()
    # client.send(now_joint)
    client.send(buf)
    print(joint)
    print('send joint')

def send_joint_loop(client: socket.socket):
    global send_continue
    while send_continue:
        joints = get_joint()
        try:
            print(joints)
            client.send(joints.to_bytes())
        except OSError:
            print('disconnected')
            break

def estimate_loop():
    global send_continue
    while send_continue:
        start = time.time()
        joints = get_joint()
        print('FPS1:{}'.format(1 / (time.time() - start)))
        handler.set_joint(joints)
        print('FPS2:{}'.format(1 / (time.time() - start)))

def streaming(client : socket.socket):
    global send_continue
    task: threading.Thread = None
    while True:
        try:
            res_id = client.recv(16)
            print('recive id:{}'.format([hex(x) for x in res_id]))
            res_id = res_id.decode()
            if(res_id == '101'):
                client.send(b'101')
                get_estimated_data(client)
            elif(res_id == '102'):
                send_joint(client)
            elif(res_id == '104'):
                send_continue = True
                task = threading.Thread(target=send_joint_loop, args=[client])
                task.setDaemon(True)
                task.start()
            elif(res_id == '105'):
                send_continue = False
            elif(res_id == '106'):
                send_continue = True
                task = threading.Thread(target=estimate_loop)
                task.setDaemon(True)
                task.start()
                buf = '106OK'.encode('utf-8')
                client.send(buf)
            else:
                break
        except ConnectionResetError:
            send_continue = False
            break
    client.close()

def loop():
    while True:
        client, address = soc.accept()
        task = threading.Thread(target=streaming, args=[client])
        task.setDaemon(True)
        task.start()
    
try:
    loop()
except KeyboardInterrupt:
    soc.close()
    sys.exit(1)
