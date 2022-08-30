import numpy as np
import math
import matplotlib.pyplot as plt

def rotate_matrix_y(theta):
    S = math.sin(theta)
    C = math.cos(theta)
    rotate_matrix = np.array([
        [ C, 0, S],
        [ 0, 1, 0],
        [-S, 0, C]
    ])
    return rotate_matrix

def rotate_matrix_z(theta):
    S = math.sin(theta)
    C = math.cos(theta)
    rotate_matrix = np.array([
        [C, -S, 0],
        [S,  C, 0],
        [0,  0, 1]
    ])
    return rotate_matrix

def rotate_z(vec:np.ndarray, theta:float) -> np.ndarray:
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

def rotate_matrix():
    vector = np.array([math.sqrt(3), 1, 1])
    angle_y = -math.atan2(vector[2], vector[0]) + math.pi / 2
    # angle_y = -1.69965976
    vector = rotate_y(vector, -angle_y)
    print('roteted_y:', vector)
    print('angle_y:', angle_y)
    angle_z = math.atan2(vector[1], vector[0])
    # angle_z = -2.37918482
    vector = rotate_z(vector, -angle_z)
    print('rotated_z:', vector)
    print('angle_z:', angle_z)

if __name__ == "__main__":
    ret = np.zeros(3)

    vector = np.array([-1, -2, -3])
    xy_pos = vector.copy()
    xy_pos[2] = 0
    pos_size = np.linalg.norm(vector)
    xy_pos_size = np.linalg.norm(xy_pos)
    if vector[2] < 0:
        ret[1] = math.acos(xy_pos_size / pos_size)
    else:
        ret[1] = -math.acos(xy_pos_size / pos_size)
    if vector[1] > 0:
        ret[2] = math.acos(vector[0] / xy_pos_size)
    else:
        ret[2] = -math.acos(vector[0] / xy_pos_size)
    
    print('ret:', ret)
    
    restore = rotate_matrix_z(ret[2]) @ rotate_matrix_y(ret[1]) @ np.array([1, 0, 0])
    print('input:', vector / pos_size)
    print('restored:', restore)
