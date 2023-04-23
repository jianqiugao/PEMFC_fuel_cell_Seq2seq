import numpy as np

from mod.遗传算法 import GA


def obj_func(data):  # 目标函数
    x = data[:, 0]
    y = data[:, 1]
    z = x ** 2 + y ** 2

    return z


import time

t1 = time.time()
ga = GA(func=lambda x: obj_func(x),
        n_dim=2, size_pop=60, max_iter=120, lb=[-1, -1], ub=[1, 1], precision=[0.00000001, 0.0001])
# 其中lb和up代表的是取值范围
best_x, best_y = ga.run()
t2 = time.time()
print(best_y, best_x, f"遗传算法用时{t2 - t1}")

x = np.arange(-1, 1, 0.0001)
y = np.arange(-1, 1, 0.001)
max_ = 0
min_x =0
min_y =0
for x_ in x:
    for y_ in y:
        z=x_**2+y_**2
        if z < max_:
            max_= z
            min_x =x_
            min_y = y_
t3 = time.time()
print(min_x,min_y,max_,f"穷举法用时{t3 - t2}")

