import math
from typing import Any
import numpy as np
from numpy import ndarray, dtype

#函数定义
def c(a,a_prime,e,q):
    c = a + e - a_prime * q
    return c

def u(c, sigma = 0.1):
    if c < 0:
        return -np.inf
    #约束：c大于等于0.
    else:
        return c**(1-sigma)/(1-sigma)
#print(u(0.5))

def grid(a_min, a_max, N):
    a_grid = np.linspace(a_min, a_max, N)
    return a_grid

def feasible(a_grid,e_grid,q):
    #输入两个数组，判断每一个数组组合是否满足令存在c>0的a'
    feas_a_grid = []
    for a in a_grid:
        feas_mark = 0
        for e in e_grid:
        #对每一个(a,e)检查。
            for a_prime in a_grid:
                if not c(a,a_prime,e,q) < 0:
                    feas_mark += 1
                    break
        if feas_mark == len(e_grid):
        # 检查完a对应的所有e，如果每个a的e都有可行c，则将a归入可行集
            feas_a_grid.append(a)
    return np.array(feas_a_grid)
#我必须要让所有的(a,e)组合，存在一个约束条件下的a'，这个a'在eh，el(即所有可能性情况下）都存在正值c。假如他只在eh的状态下存在正值c，而被纳入考量。那我如果这期a在遍历a'计算下期V值的时候，发现这个V(a')里头包含的u(c)不管怎么样都到不了正值，进而产生负无穷。在求maxV()时会出错。

def feas_close(origin_a_grid,q):
#给我原始a_grid，反复用feasible函数求可行集直到闭。
    a_old = origin_a_grid
    while True:
        a_new = feasible(a_old, e_grid,q)
        if np.array_equal(a_new,a_old):
            break
        a_old = a_new
    return a_new

def T(V0,q,a_grid,e_grid):
#V1采用矩阵形式便于后续计算。但是T算子输入矩阵输出矩阵。
    V1 = np.zeros((len(a_grid),len(e_grid)))
    policy = np.zeros((len(a_grid),len(e_grid)))
    #V1相当于一张函数表。输入初始状态(a,e),返回对应位置的最优值。
    for ia,a in enumerate(a_grid):
        for ie,e in enumerate(e_grid):
            #开始对于每个值求解
            #V1的作用是遍历(a,e)，找到满足最大效用的下期资产a'*。V1需要输入当期资产a。V1还需要知道当期e，这样求期望知道引用哪部分。
            values = []
            #values进入函数。因为在第一次定义，比如V1=T(V0)，后，引用V1就会直接从def V1引用。
            for ia_prime, a_prime in enumerate(a_grid):
            #enumerate返回tuple[index, number]位置和值都返回。
                EV = 0
                #全局变量解决遍历问题
                for ie_prime, e_prime in enumerate(e_grid):
                    #负责求期望
                    EV += (
                            P_trans_matrix[ie_prime,ie]*V0[ia_prime,ie_prime]
                    )
                values.append(u(c(a,a_prime,e,q)) + beta * EV)
                #####@@@###u函数实现c>0约束。但是我害怕对于某个(a,e)，所有a'都令c<0，最终导致V(a,e)是-inf。
                #对于每一个下一期a'求值,values一共有len(grid_a)个值。
            V1[ia,ie] = max(values)
            policy[ia,ie] = a_grid[np.argmax(values)]
            #为什么返回值不能直接用到下一期？这个a只针对这个输入V0，下一期迭代V0不同，最优选择可能不同。而下一期计算的时候我们不需要知道导致V0的最大值V，只需要知道V0的最大返回值即可。
            #policy返回使v1总效用最大的下期政策函数a'。
    #V函数输入下期值，返回最优效用
    return V1, policy
#print(grid()) grid通过。

#a是一个离散数组，e是状态向量。
def Bellman(a_grid,e_grid,q):
    policy_new = np.zeros((len(a_grid),len(e_grid)))
#Bellman返回家庭的最优化效用政策函数T。
    epsilon = 1e-6
#Bellman函数对于每一个给定的(a,e)组合，进行V的递归求解。
    V0 = np.zeros((len(a_grid),len(e_grid)))
    V_old = V0
    while True:
    #循环找最优V*
        diff = np.zeros((len(a_grid),len(e_grid)))
        V_new, policy_new =T(V_old,q,a_grid,e_grid)
        diff = np.abs(V_new - V_old)
        #遍历所有取最大，看||V_new - V_old||是否符合定义。
        error = np.max(diff)
        #npmax和普通max不一样。np.max将矩阵拆分为元素再取max，但是python的max会将矩阵看成两层行数组，会在每列取最大，然后生成新数组。
        if error < epsilon:
            break
        V_old = V_new
    return V_new, policy_new
#政策函数及V的结构：[[(a0,eh),[a0,el)]....]以a为界分为一级数组。以e再分二级数组。eh对应的在前头。

def prob_matrix_func(a_grid, e_grid, policy, P_trans_matrix):
#概率矩阵完了还要求分布fai。但是注意，概率矩阵每列加和是1。连续情况考虑上界boundary。
    prob_matrix = np.zeros((len(a_grid)*len(e_grid),len(a_grid)*len(e_grid)))
    for ord,a_prime in enumerate(policy):
    #第ord个状态，会转到下一期的a'的对应表格。
        a_prime = a_prime[0]
        #a是一个矩阵的一行，即一维向量[x]，sorted返回的是array([x])，因为searchsorted可以接触多组数，返回一个包含多个位置的数组。但我们目前只用一个。所以就不用向量即可。
        index = np.searchsorted(a_grid,a_prime)
        if index == 0:
            #index0说明到了a'的最小边界。下一期a'确定最小。ord%2确定当前状态后，下一期之可能有两个状态。因为a'确定，也就只有两个可能。
            prob_matrix[0, ord] = P_trans_matrix[0, ord % 2]
            prob_matrix[1, ord] = P_trans_matrix[1, ord % 2]
        #在a_grid[index-1]和a_grid[index]之间。
        else:
            a_small_prob = (a_grid[index]-a_prime)/(a_grid[index]-a_grid[index-1])
            #注意检验grid端点：python采用left，所以[0,1][1,2]区间，分界点1插在第[1]个位置（前头还有0）所以index返回0。
            a_big_prob = 1 - a_small_prob
        #这里刚分完a。还要接下来按照eh和el分。
            prob_matrix[2 * (index - 1) + 0, ord] = a_small_prob * P_trans_matrix[0, ord % 2]
        #从ord状态，转到index-1（也就是a'分类后较小值）的+0（eh）状态的概率:a_xiao（a较小值的概率）*ord状态对应的下一期eh的概率。ord%2,余0说明当期高状态，对应第一列。
            prob_matrix[2 * (index - 1) + 1, ord] = a_small_prob * P_trans_matrix[1, ord % 2]
            prob_matrix[2 * (index) + 0, ord] = a_big_prob * P_trans_matrix[0, ord % 2]
            prob_matrix[2 * (index) + 1, ord] = a_big_prob * P_trans_matrix[1, ord % 2]
    return prob_matrix

def Stab_Distri(phi,prob_matrix):
    phi_old = phi
    epsilon = 1e-6
    phi_new = np.zeros((len(phi_old),1))
    diff = np.zeros((len(phi_old), 1))
    while True:
        phi_new = prob_matrix @ phi_old
        diff = np.abs(phi_new - phi_old)
        error = np.max(diff)
        #print(phi_new)
        if error < epsilon:
            break
        phi_old = phi_new
    return phi_new

def a_judge(a_grid,e_grid,q,P_trans_matrix):
    # print(a_grid)
    phi = (1 / (2 * len(a_grid))) * np.ones([2 * len(a_grid), 1])
    # print(phi)
    V, policy350 = Bellman(a_grid, e_grid, q)
    policy700 = policy350.reshape(-1, 1)
    # 700x700的概率矩阵。将mu1分布转化为mu2分布，第一行意义：每一个状态转化到（a0，eh）（第一个状态的概率）
    prob_matrix = prob_matrix_func(a_grid, e_grid, policy700, P_trans_matrix)
    # print(prob_matrix)
    phi_stable = Stab_Distri(phi, prob_matrix)
    # print(phi_stable,phi_stable.sum(),np.max(abs(prob_matrix @ phi_stable - phi_stable)))
    # print(prob_matrix.sum(axis=0))
    a_phi_stable = np.zeros((len(a_grid), 1))
    for i in range(len(phi_stable)):
        a_phi_stable[i // 2] += phi_stable[i, 0]
        # phi 350x1 a_phi_stable 是资产分布。
    a_sum_stable = a_grid.T @ a_phi_stable
    return a_sum_stable,a_phi_stable

a_min = -2
a_max = 5
N = 300
beta = 0.5
e_h = 1
e_l = 0.1
e_grid = np.array([e_h,e_l])
origin_a_grid = grid(a_min, a_max, N)
#先高再低符合结构
P_trans_matrix = np.array([
    [0.925,0.5],
    #ij表示j转换到i的概率
    [0.075,0.5]
])

q_low = 0
q_high = 1
a_grid_low = feas_close(origin_a_grid, q_low)
a_grid_high = feas_close(origin_a_grid, q_high)
a_sum_low = a_judge(a_grid_low, e_grid, q_low, P_trans_matrix)[0]
a_sum_high = a_judge(a_grid_high, e_grid, q_high, P_trans_matrix)[0]
if  a_sum_low * a_sum_high > 0:
    print("端点同号")
a_sum_old = 0
a_sum_min = 100
q_min = 0

while True:
    q_med = (q_low + q_high) / 2
    print(q_low,q_high,q_med)
    a_grid = feas_close(origin_a_grid, q_med)
    #print(a_grid)
    a_sum,a_phi_stable= a_judge(a_grid,e_grid,q_med,P_trans_matrix)
    print(a_sum)

    if abs(a_sum) <= abs(a_sum_min):
        a_sum_min = abs(a_sum)
        q_min = q_med
        a_grid_min = a_grid.copy()
        a_phi_stable_min = a_phi_stable.copy()
    #数值最小化输出

    if abs(a_sum) < 1e-6:
        print(q_med,"一直到均衡q值")
        break
    elif abs(q_high-q_low) < 1e-10:
        print(q_min,a_sum_min,"未找到足够均衡q值，但区间足够已足够细分")
        break
    #数值判断。

    if a_sum > 0:
        #中位置资产大于0，资产供给过剩，降低利率，提高q，减少需求。如果过热，q往上区间走。
        q_low = q_med
    elif a_sum < 0:
        q_high = q_med
    else:
        print(a_sum,"资产总和非正非负非0，error")
        break
    #数值更新

positive_assets = 0.0
negative_assets = 0.0

for ia, a in enumerate(a_grid):
    if a > 0:
        positive_assets += a * a_phi_stable[ia, 0]
    elif a < 0:
        negative_assets += (-a) * a_phi_stable[ia, 0]

market_residual = positive_assets - negative_assets

gross_positions = positive_assets + negative_assets

relative_error = (
    abs(market_residual) / gross_positions
    if gross_positions > 0
    else 0.0
)

print("正资产 L =", positive_assets)
print("债务 B =", negative_assets)
print("市场残差 L-B =", market_residual)
print("相对误差 =", relative_error)

"""
#算子定义
def T(V0):
#a是一个数组，作遍历用。V对所有的(a,e)都能返回值。
    def V1[ia,ie]:
        []
        #索引方便定位转换矩阵中P的位置。
        output = []
        for ia,a in enumerate(a_grid):
            for ie,e in enumerate(e_grid):
    #V1的作用是通过遍历a_grid中各值，找到满足最大效用的下期资产a'*。V1需要输入当期资产a。V1还需要知道当期e，这样求期望知道引用哪部分。
                values = []
                #values进入函数。因为在第一次定义，比如V1=T(V0)，后，引用V1就会直接从def V1引用。
                for ia_prime, a_prime in enumerate(a_grid):
                    #enumerate返回tuple[index, number]位置和值都返回。
                    EV = 0
                #全局变量解决遍历问题
                    for ie_prime, e_prime in enumerate(e_grid):
                        EV += (
                                P_trans_matrix[ie_prime,ie]*V0(ia_prime,ie_prime)
                        )
                    values.append(u(c(a,a_prime,e)) + beta * EV)
                output.append(max(values))
        return output
    #V函数输入下期值，返回最优效用
    return V1
#print(grid()) grid通过。
"""





