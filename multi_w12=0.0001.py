import os
import sys
sys.path.append(r"../")
import numpy as np
import matplotlib.pyplot as plt
# 节点
from HH import HH
# from FHN import FHN
# 导入突触
from synapse_alpha import synbasedelay, synbasedelay_12
# 拓扑结构
from ws_small_world import create_sw
# 计算，统计量
from statis import cal_synFactor   # , cal_synEuclidean
from utils import delayer, noise_types
from utils_plot import plot_firing_raster

Q1 = int(sys.argv[1])

w12 = .0001
wmin = float(sys.argv[2])
wmax = float(sys.argv[3])
filename = r'multiplex_Q1='+str(Q1)+'_w12='+str(w12)+'_wmin'+str(wmin)+'-wmax'+str(wmax)+'.txt' 
# filename = r'multiplex_H-M_w12='+str(w12)+'_tau12='+str(tau12)+'.txt'   # 
print(filename)
f = open(filename,'w')


N1 = 200 # 第一层神经元数量
N2 = 200 # 第二层神经元数量

dt = 0.01
method = "euler"
synType = "electr"

Tn = 100000
R_th = .9

Q2 = Q1+40

# L
k1 = 4
p1 = 0.5
F1 = .001

k2 = 20
p2 = 0.5
F2 = 1.
w2 = 0.01

w1 = wmin
while w1 < wmax:

    q = 0
    for count in range(Q1, Q2, 1):
        np.random.seed(count)

        delayer1 = delayer(N1, 0)
        delayer2 = delayer(N1, 0)
        delayer12 = delayer(N1, 0)
        delayer21 = delayer(N1, 0)

        # 第一层
        nodes1 = HH(N=N1, method=method, dt=dt)
        nodes1.Iex = 10
        # BS初始值
        nodes1.mem = np.random.uniform(-80, 30, N1)
        nodes1.m = np.random.uniform(0, 1, N1)
        nodes1.n = np.random.uniform(0, 1, N1)
        nodes1.h = np.random.uniform(0, 1, N1)

        # 第二层
        nodes2 = HH(N=N2, method=method, dt=dt)
        nodes2.Iex = 10
        # BS初始值
        nodes2.mem = np.random.uniform(-80, 30, N2)
        nodes2.m = np.random.uniform(0, 1, N2)
        nodes2.n = np.random.uniform(0, 1, N2)
        nodes2.h = np.random.uniform(0, 1, N2)

        # 创建突触连接
        # Conn = create_sw()
        # 1 --> 1
        Conn11 = create_sw(N1, k1, p1)
        syn11 = synbasedelay(nodes1, nodes1, Conn11, synType, delayer1)
        syn11.w.fill(w1)
        syn11.F = F1
        syn11.p = p1

        # 2 --> 2
        Conn22 = create_sw(N2, k2, p2)
        # post_degree2 = Conn22.sum(1)
        syn22 = synbasedelay(nodes2, nodes2, Conn22, synType, delayer2)
        syn22.w.fill(w2)
        syn22.F = F2
        syn22.p = p2

        # 1 --> 2
        # 0维度--post，1维度--pre
        Conn12 = np.eye(N2, N1)
        syn12 = synbasedelay_12(nodes1, nodes2, Conn12, synType, delayer12)
        syn22.w.fill(w12)

        # 2 --> 1
        Conn21 = np.eye(N1, N2)
        syn21 = synbasedelay_12(nodes2, nodes1, Conn21, synType, delayer21)
        syn22.w.fill(w12)

        R_cal = cal_synFactor(Tn, N1)

        # 初始化
        for i in range(50000):
            nodes1()  
            nodes2()  

        # 加入突触初始化
        for i in range(200000):
            # 节点群1接受到的突触电流
            I_syn11 = syn11()
            I_syn21 = syn21()
            # 节点群2接受到的突触电流
            I_syn12 = syn12()
            I_syn22 = syn22()

            post_degree1 = syn11.conn.sum(1)
            post_degree2 = syn22.conn.sum(1)

            # 突触电流
            I1 = (I_syn11/post_degree1)+I_syn21 
            nodes1(I1) 

            I2 = (I_syn22/post_degree2)+I_syn12
            nodes2(I2)  


        for i in range(Tn):
            # 节点群1接受到的突触电流
            I_syn11 = syn11()
            I_syn21 = syn21()
            # 节点群2接受到的突触电流
            I_syn12 = syn12()
            I_syn22 = syn22()

            post_degree1 = syn11.conn.sum(1)
            post_degree2 = syn22.conn.sum(1)

            # 突触电流
            I1 = (I_syn11/post_degree1)+I_syn21 
            nodes1(I1) 

            I2 = (I_syn22/post_degree2)+I_syn12
            nodes2(I2)  

            R_cal(nodes1.mem)

        R = R_cal.return_syn()

        if R > R_th:
            q += 1

    f.write(f"{w1} {count} {q}\n")
    print(w1, count, q)

    w1 += .0004

f.close() 