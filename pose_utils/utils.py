from aifc import _aifc_params
from email import utils
import enum
import math
from queue import Queue
from tkinter.tix import MAX
import numpy as np
import time
import struct
import logging
import heapq
from pose_utils.config import Config, InterpolationType

logging.basicConfig(format='[%(levelname)s] %(asctime)s: %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

class JointInfo:
    """関節情報を取り扱うクラス

    Attributes
    ----------
    joints : np.ndarray
        各関節のベクトル情報
    root_pos : np.ndarray
        人物の中心位置
    root_rot : np.ndarray
        人物の向き
        
    Notes
    -----
    簡単のために、ほとんどの場合の人数は過不足にかかわらず `MAX_PEOPLE` に統一される。
    """
    
    FORMAT_VERSION = 0x01
    NUM_JOINT = 21
    MAX_PEOPLE = 5
    
    def __init__(self, joints: np.ndarray=np.zeros((MAX_PEOPLE, NUM_JOINT, 3)), 
            root_pos: np.ndarray=np.zeros((MAX_PEOPLE, 3)), root_rot: np.ndarray=np.zeros((MAX_PEOPLE,))) -> None:
        """コンストラクタ

        Parameters
        ----------
        joints : np.ndarray, optional
            各関節のベクトル情報, by default np.zeros((MAX_PEOPLE, NUM_JOINT, 3))
        root_pos : np.ndarray, optional
            人物の中心位置, by default np.zeros((MAX_PEOPLE, 3))
        root_rot : np.ndarray, optional
            人物の向き, by default np.zeros((MAX_PEOPLE,))
        """
        self.joints = joints.copy()
        self.joints.resize(self.MAX_PEOPLE, self.NUM_JOINT, 3, refcheck=False)
        self.root_pos = root_pos.copy()
        self.root_pos.resize(self.MAX_PEOPLE, 3, refcheck=False)
        self.root_rot = root_rot.copy()
        self.root_rot.resize(self.MAX_PEOPLE, refcheck=False)
    
    def check_init(self):
        assert self.MAX_PEOPLE == len(self.joints)
        assert self.MAX_PEOPLE == len(self.root_pos)
        assert self.MAX_PEOPLE == len(self.root_rot)
    
    def to_bytes(self) -> bytes:
        """関節情報をバイト列に変換

        Returns
        -------
        bytes
            関節情報を表すバイト列
        """
        self.check_init()
        ret = struct.pack('<b', self.FORMAT_VERSION)
        ret += struct.pack('<b', self.MAX_PEOPLE)
        for i in range(self.MAX_PEOPLE):
            for j in range(self.NUM_JOINT):
                ret += struct.pack('<f', self.joints[i][j][0])
                ret += struct.pack('<f', self.joints[i][j][1])
                ret += struct.pack('<f', self.joints[i][j][2])
            ret += struct.pack('<f', self.root_pos[i][0])
            ret += struct.pack('<f', self.root_pos[i][1])
            ret += struct.pack('<f', self.root_pos[i][2])
            ret += struct.pack('<f', self.root_rot[i])
            # print(struct.pack('<f', self.root_rot[i]).hex())
        # print(ret.hex())
        return ret
    
    @staticmethod
    def parse_bytes1(byte: bytes):
        num_joint = 21
        unit_byte = (num_joint * 3 + 3 + 1) * 4
        
        num_person = byte[1]
        joints = [np.zeros((num_joint, 3)) for _ in range(num_person)]
        root_pos = [np.zeros(3) for _ in range(num_person)]
        root_rot = [0 for _ in range(num_person)]
        
        for i in range(num_person):
            jinfo = byte[2 + i * unit_byte: 2 + (i + 1) * unit_byte]
            logger.debug('unit:{} jinfo shape:{}'.format(unit_byte, len(jinfo)))
            p = 0
            for j in range(num_joint):
                joints[i][j] = [struct.unpack('<f', jinfo[p + x * 4 : p + (x + 1) * 4])[0] for x in range(3)]
                p += 4 * 3
            root_pos[i] = [struct.unpack('<f', jinfo[p + x * 4 : p + (x + 1) * 4])[0] for x in range(3)]
            p += 3 * 4
            root_rot[i] = struct.unpack('<f', jinfo[p : p + 4])[0]
        
        ret = JointInfo(joints=np.array(joints), root_pos=np.array(root_pos), root_rot=np.array(root_rot))
        return ret
    
    @staticmethod
    def from_bytes(byte: bytes):
        """バイト列から関節情報を生成

        Parameters
        ----------
        byte : bytes
            関節情報を表すバイト列

        Returns
        -------
        JointInfo
            解析された関節情報
            
        See Also
        --------
        フォーマットバージョンが0x01の場合は関数 `parse_bytes1` を参照
        """
        format_ver = byte[0]
        if format_ver == 0x01:
            ret = JointInfo.parse_bytes1(byte)
        return ret

    def __str__(self):
        joints = self.joints.astype('str')
        root_pos = self.root_pos.astype('str')
        root_rot = self.root_rot.astype('str')
        
        joints_str = [[' '.join(y) for y in x] for x in joints]
        joints_str = [','.join(x) for x in joints_str]
        joints_str = ';'.join(joints_str)
        pos_str = [' '.join(x) for x in root_pos]
        pos_str = ';'.join(pos_str)
        rot_str = ';'.join(root_rot)
        
        return '{}/{}/{}'.format(joints_str, pos_str, rot_str)
    
    def __repr__(self) -> str:
        return 'JointInfo({})'.format(self.__str__())
        
    def __add__(self, other):
        if type(other) != JointInfo: raise TypeError()
        
        joint = self.joints + other.joints
        rootPos = self.root_pos + other.root_pos
        rootRot = self.root_rot + other.root_rot
        return JointInfo(joint, rootPos, rootRot)
    
    def __mul__(self, other: float):
        if type(other) != float and type(other) != int: raise TypeError()
        
        joint = self.joints * other
        rootPos = self.root_pos * other
        rootRot = self.root_rot * other
        return JointInfo(joint, rootPos, rootRot)
    
    def __rmul__(self, other: float):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + -other
    
    def __truediv__(self, other: float):
        return self * (1 / other)

class InterpolationFunction:
    """補間関数を生成するクラス
    
    Attributes
    ----------
    acceleration : JointInfo
        計算に用いる関節の加速度
    velocity : JointInfo
        計算に用いる関節の速度
    """
    
    k = 4e-2
    
    def __init__(self, acceleration: JointInfo, velocity: JointInfo):
        """コンストラクタ

        Parameters
        ----------
        acceleration : JointInfo
            加速度
        velocity : JointInfo
            初速度
        """
        self.acceleration = acceleration
        self.velocity = velocity
        
    def getVelocity(self, span):
        """速度を返す関数

        Parameters
        ----------
        span : float
            返す速度の秒数

        Returns
        -------
        JointInfo
            速度
        """
        if(Config.interpolationType == InterpolationType.ACCELERATION):
            ### 等加速度補間
            ret = self.velocity + span * self.acceleration
            
        elif(Config.interpolationType == InterpolationType.RESISTANCE):
            ### 抵抗付き等加速度補間
            k = InterpolationFunction.k
            a_per_k = self.acceleration / k
            ret = (self.velocity - a_per_k) * math.exp(-k * span) + a_per_k
        
        elif(Config.interpolationType == InterpolationType.LINER):
            ### 線形補間
            ret = self.velocity
        
        else:
            ret = JointInfo()
        
        return ret
    
    def getDisplacement(self, span):
        """変位を返す関数

        Parameters
        ----------
        span : float
            返す変位の秒数

        Returns
        -------
        JointInfo
            変位
        """
        if(Config.interpolationType == InterpolationType.ACCELERATION):
            ### 等加速度補間
            ret = self.velocity * span + self.acceleration * (span * span / 2)
        elif(Config.interpolationType == InterpolationType.RESISTANCE):
            ### 抵抗付き等加速度補間
            k = InterpolationFunction.k
            ret = (self.acceleration * span - (self.velocity - self.acceleration / k) * (math.exp(-k * span) - 1)) / k
        elif(Config.interpolationType == InterpolationType.LINER):
            ### 線形補間
            ret = self.velocity * span
        else:
            ret = JointInfo()
        
        return ret

class JointVelocity:
    """関節の速度と加速度を持つクラス
    
    Attributes
    ----------
    t1 : float
        移動時間
    time : float
        インスタンスが作成された時刻
    velocity : JointInfo
        関節の(初)速度
    acceleration : JointInfo
        関節の加速度
    normalFunction : InterpolationFunction
        通常時に用いる関数
    velocityLast : JointInfo
        `t1` 秒後の速度
    displacementLast : JointInfo
        `t1` 秒後の変位
    """
    def __init__(self, initialVelocity: JointInfo = JointInfo(), acceleration: JointInfo = JointInfo(), t1: float = 0) -> None:
        """コンストラクタ

        Parameters
        ----------
        initialVelocity : JointInfo, optional
            関節の(初)速度, by default JointInfo()
        acceleration : JointInfo, optional
            関節の加速度, by default JointInfo()
        t1 : float, optional
            移動時間, by default 0
        """
        self.t1 = t1
        self.time = time.time()
        self.velocity = initialVelocity
        self.acceleration = acceleration
        
        self.normalFunction = InterpolationFunction(self.acceleration, self.velocity)
        self.velocityLast = self.getVelocityAt(t1)
        self.displacementLast = self.getDisplacementAt(t1)
    
    def getVelocityAt(self, span) -> JointInfo:
        """指定した時間の速度を取得

        Parameters
        ----------
        span : float
            時間

        Returns
        -------
        JointInfo
            取得した速度
        """
        if(span <= self.t1):
            ret = self.normalFunction.getVelocity(span)
        elif(Config.isPowerReleaseMode):
            ### リリース
            ret = InterpolationFunction(JointInfo(), self.velocityLast).getVelocity(span - self.t1)
        else:
            ### 停止
            ret = JointInfo()
        
        return ret
        
    def getDisplacementAt(self, span) -> JointInfo:
        """指定した時間の変位を取得

        Parameters
        ----------
        span : float
            時間

        Returns
        -------
        JointInfo
            取得した変位
        """
        if(span <= self.t1):
            ret = self.normalFunction.getDisplacement(span)
        elif(Config.isPowerReleaseMode):
            ### リリース
            ret = InterpolationFunction(JointInfo(), self.velocityLast).getDisplacement(span - self.t1) + self.displacementLast
        else:
            ### 停止
            ret = self.displacementLast
        
        return ret
    
    @staticmethod
    def getAcceleration(delta: JointInfo, initialVelocity: JointInfo, duration: float) -> JointInfo:
        """条件から加速度を計算

        Parameters
        ----------
        delta : JointInfo
            フレーム間の変位の差
        initialVelocity : JointInfo
            関節の初速度
        duration : float
            移動時間

        Returns
        -------
        JointInfo
            加速度
        """
        if(Config.interpolationType == InterpolationType.ACCELERATION):
            ### 等加速度補間
            acceleration = 2 * (delta - initialVelocity * duration) / duration ** 2
            
        elif(Config.interpolationType == InterpolationType.RESISTANCE):
            ### 抵抗付き等加速度補間
            k = InterpolationFunction.k
            expm1_mkt = math.expm1(-k * duration)
            acceleration = (delta * k + initialVelocity * expm1_mkt) / (duration + expm1_mkt / k)
            
        else:
            ### 線形補間 補間なし
            acceleration = JointInfo()
            
        return acceleration
    
class Edge:
    """ネットワークフロー用の構造体
    """
    def __init__(self, to, cap, cost, rev):
        self.to = to
        self.cap = cap
        self.cost = cost
        self.rev = rev

class JointHandler:
    """関節の処理を行うクラス
    
    Attributes
    ----------
    joints : JointInfo
        最新フレームの関節位置
    velocity : JointInfo
        最新フレームの速度
    """
    DISPLACEMENT_SKIP_THRESHOLD = 1000
    
    def __init__(self) -> None:
        """コンストラクタ
        """
        self.joints = JointInfo()
        self.velocity = JointVelocity()
        
    def get_joint(self, cur_time=None) -> JointInfo:
        """関節を処理して返す

        Parameters
        ----------
        cur_time : float, optional
            取得する関節の時刻, by default None

        Returns
        -------
        JointInfo
            処理した関節情報
        """
        if cur_time is None:
            cur_time = time.time()
        t = cur_time - self.velocity.time
        
        # new_joint = []
        # new_pos = []
        # new_rot = []
        # # delta_time = self.joints[1].time - self.joints[0].time
        # liner_vel = self.joints[0].velocity * t
        # for i in range(JointInfo.MAX_PEOPLE):
        #     joint = np.zeros((JointInfo.NUM_JOINT, 3))
        #     for j in range(JointInfo.NUM_JOINT):
        #         # delta = self.joints[1].joints[i][j] - self.joints[0].joints[i][j]
        #         # alpha = 2 * (delta - liner_vel) / delta_time * delta_time
        #         joint[j] = liner_vel[i][j] + self.alpha[i][j] * t * t / 2 + self.joints[0].joints[i][j]
        #     new_joint.append(joint)
            
        #     velocity = (self.joints[1].root_pos[i] - self.joints[0].root_pos[i]) / self.joints[1].time
        #     root_pos = velocity * t + self.joints[0].root_pos[i]
        #     new_pos.append(root_pos)
            
        #     velocity = (self.joints[1].root_rot[i] - self.joints[0].root_pos[i]) / self.joints[1].time
        #     root_rot = velocity * t + self.joints[0].root_rot[i]
        #     new_rot.append(root_rot)
        
        
        ret = self.velocity.getDisplacementAt(t) + self.joints
        return ret
        # return JointInfo(np.array(new_joint), np.array(new_pos), np.array(new_rot))
    
    def swapJoint(self, oldJoint: JointInfo, newJoint: JointInfo):
        """人物対応に基づく関節の入れ替え

        Parameters
        ----------
        oldJoint : JointInfo
            前フレームの関節位置
        newJoint : JointInfo
            現在フレームの関節位置

        Returns
        -------
        JointInfo
            入れ替え後の関節位置

        Raises
        ------
        Exception
            グラフ作成時のエラー
        """
        oldRootPos = oldJoint.root_pos
        newRootPos = newJoint.root_pos
        
        numPerson = newJoint.MAX_PEOPLE
        numVertex = numPerson * 2 + 2
        
        graph = [[] for _ in range(numVertex)]
        
        for i in range(numPerson):
            vfrom = 0
            vto = i + 1
            graph[vfrom].append(Edge(vto, 1, 1, len(graph[vto])))
            graph[vto].append(Edge(vfrom, 0, -1, len(graph[vfrom]) - 1))
        
        for i in range(numPerson):
            for j in range(numPerson):
                vfrom = i + 1
                vto = j + numPerson + 1
                
                cost = np.linalg.norm(oldRootPos[i] - newRootPos[j])
                # cost = 0
                # for k in range(JointInfo.NUM_JOINT):
                #     cost += np.linalg.norm(oldJoint.joints[i][k] - newJoint.joints[j][k])
                
                graph[vfrom].append(Edge(vto, 1, cost, len(graph[vto])))
                graph[vto].append(Edge(vfrom, 0, -cost, len(graph[vfrom]) - 1))
        
        for i in range(numPerson + 1, numPerson * 2 + 1):
            vfrom = i
            vto = numVertex - 1
            graph[vfrom].append(Edge(vto, 1, 1, len(graph[vto])))
            graph[vto].append(Edge(vfrom, 0, -1, len(graph[vfrom]) - 1))
        
        graph, res = minCostFlow(graph, numPerson)
        if res == -1: raise Exception()
        
        oldNewPair = [-1] * numPerson
        
        for i in range(numPerson):
            gindex = i + 1
            for j in range(len(graph[gindex])):
                if(graph[gindex][j].cap == 0):
                    oldNewPair[i] = graph[gindex][j].to - numPerson - 1
                    break
        
        retJoint = [[]] * numPerson
        retRootPos = [[]] * numPerson
        retRootRot = [[]] * numPerson
        
        # print("cost:{}".format(res))
        # for i, g in enumerate(graph):
        #     for e in g:
        #         if(i < e.to and e.cap == 0):
        #             print(i, e.to, e.cost)
        
        
        for i in range(numPerson):
            to = oldNewPair[i]
            retJoint[i] = newJoint.joints[to]
            retRootPos[i] = newJoint.root_pos[to]
            retRootRot[i] = newJoint.root_rot[to]
        
        return JointInfo(np.array(retJoint), np.array(retRootPos), np.array(retRootRot))
    
    def rotateY(self, deg):
        """Y軸回転の3次元回転行列を取得

        Parameters
        ----------
        deg : float
            角度(ラジアン)

        Returns
        -------
        np.ndarray
            回転行列
        """
        C = np.cos(deg)
        S = np.sin(deg)
        
        Ry = np.matrix((
            ( C,  0,  S),
            ( 0,  1,  0),
            (-S,  0,  C)
        ))
        return Ry
    
    def convertToRelative(self, jointInfo: JointInfo):
        """関節を相対座標に変換

        Parameters
        ----------
        jointInfo : JointInfo
            変換する関節

        Returns
        -------
        JointInfo
            変換後の関節
        """
        joints = jointInfo.joints
        num_joint = JointInfo.NUM_JOINT
        num_person = len(joints)
        rel_vec = np.zeros((num_person, num_joint, 3))
        root_joint = 14

        root = [16, 15, 1, 2, 3, 1, 5, 6, 14, 8, 9, 14, 11, 12, -1, 14, 1, 4, 7, 10, 13]
        tree = [[] for _ in range(num_joint)]
        for i, j in enumerate(root):
            if j == -1: continue
            tree[j].append(i)
        
        for k, joint in enumerate(joints):
            Ry = self.rotateY(-jointInfo.root_rot[k])
            
            # Calcurate absolution angle using BFS
            que = Queue()
            que.put(root_joint)
            while not que.empty():
                idx = que.get()
                if idx != root_joint:
                    if((joint[idx] == [0, 0, 0]).all() or (joint[root[idx]] == [0, 0, 0]).all()):
                        rel_vec[k][idx] = [0, 0, 0]
                    else:
                        rel_pos = joint[idx] - joint[root[idx]]
                        rel_vec[k][idx] = np.dot(Ry, rel_pos)
                    
                for j in tree[idx]:
                    que.put(j)
                    
        return JointInfo(rel_vec, jointInfo.root_pos, jointInfo.root_rot)
    
    def set_joint(self, joints: JointInfo):
        """関節を更新

        Parameters
        ----------
        joints : JointInfo
            新しい関節
        """
        # pre_roots = self.joints[0].root_pos
        # post_roots = joints.root_pos
        # joint_v = JointVelocity()
        
        # used = set()
        # for i in range(JointInfo.MAX_PEOPLE):
        #     m = 1e9
        #     nearest = None
        #     for j in range(JointInfo.MAX_PEOPLE):
        #         if j in used: continue
        #         norm = np.linalg.norm(post_roots[i] - pre_roots[j])
        #         if m > norm:
        #             m = norm
        #             nearest = j
        #     used.add(nearest)
        #     joint_v.joints[nearest] = joints.joints[i]
        #     joint_v.root_pos[nearest] = joints.root_pos[i]
        #     joint_v.root_rot[nearest] = joints.root_rot[i]
        
        if Config.isSwapAvailable:
            joints = self.swapJoint(self.joints, joints)
        relativeJoints = self.convertToRelative(joints)
        
        # joints.__class__ = JointVelocity
        # joint_v.velocity = np.zeros((JointInfo.MAX_PEOPLE, JointInfo.NUM_JOINT, 3))
        # joint_v.time = time.time()
        # self.joints[1] = joint_v
        # self.joints[1].time = time.time()
        
        
        duration = time.time() - self.velocity.time #+ 0.05
        self.joints += self.velocity.getDisplacementAt(duration)
        delta: JointInfo = relativeJoints - self.joints
        
        if(Config.interpolationType == InterpolationType.ACCELERATION or Config.interpolationType == InterpolationType.RESISTANCE):
            ### 等加速度補間
            initialVelocity = self.velocity.getVelocityAt(duration)
        elif(Config.interpolationType == InterpolationType.LINER):
            ### 線形補間
            initialVelocity = delta / duration
        else:
            ### 補間なし
            initialVelocity = JointInfo()
            self.joints = relativeJoints
                
        if(np.amax(delta.root_pos[:, 2]) < JointHandler.DISPLACEMENT_SKIP_THRESHOLD):
            acceleration = JointVelocity.getAcceleration(delta, initialVelocity, duration)
        else:
            acceleration = JointInfo()
            initialVelocity = JointInfo()
            self.joints = relativeJoints
        
        
        self.velocity = JointVelocity(initialVelocity, acceleration, duration)
        
        # assert self.velocity.getDisplacementAt(duration) + relativeJoints
        
        # delta_time = self.joints[1].time - self.joints[0].time
        # liner_vel = self.joints[0].velocity * delta_time
        # theta = np.array(self.joints[1].joints)
        # self.alpha = 2 * (theta - liner_vel) / delta_time * delta_time

INF = 1e9

def minCostFlow(g, flow):
    """最小費用流問題を解く

    Parameters
    ----------
    g : array_like
        グラフ(隣接リスト)
    flow : int
        流量

    Returns
    -------
    g : array_like
        フローを流した後のグラフ
    res : int
        最小費用
    """
    vertexSize = len(g)
    res = 0
    h = [0] * vertexSize
    prevv = [0] * vertexSize
    preve = [0] * vertexSize
    
    while(flow > 0):
        q = []
        d = [INF] * vertexSize
        d[0] = 0
        heapq.heappush(q, (0, 0))
        
        while(len(q) != 0):
            p = heapq.heappop(q)
            v = p[1]
            if(d[v] < p[0]): continue
            
            for i in range(len(g[v])):
                e: Edge = g[v][i]
                if(e.cap > 0 and d[e.to] > d[v] + e.cost + h[v] - h[e.to]):
                    d[e.to] = d[v] + e.cost + h[v] - h[e.to]
                    prevv[e.to] = v
                    preve[e.to] = i
                    heapq.heappush(q, (d[e.to], e.to))
        
        if(d[vertexSize - 1] == INF):
            return g, -1
        
        for v in range(vertexSize):
            h[v] += d[v]
        
        df = flow
        vi = vertexSize - 1
        while(vi != 0):
            df = min(df, g[prevv[vi]][preve[vi]].cap)
            vi = prevv[vi]
        
        flow -= df
        res += df * h[vertexSize - 1]
        
        vi = vertexSize - 1
        while(vi != 0):
            e1: Edge = g[prevv[vi]][preve[vi]]
            e1.cap -= df
            g[prevv[vi]][preve[vi]] = e1
            
            e2: Edge = g[vi][e1.rev]
            e2.cap += df
            g[vi][e1.rev] = e2
            
            vi = prevv[vi]
    return g, res
