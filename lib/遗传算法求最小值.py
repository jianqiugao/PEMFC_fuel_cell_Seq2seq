import numpy as np
from matplotlib import pyplot as plt

from mod.遗传算法 import GA


def obj_func(data):  # 目标函数
    # print(data.shape)
    x = data[:, 0]
    y = data[:, 1]
    z = x ** 2 + y ** 2
    print(isinstance(z,np.ndarray),z.shape)
    return z


ga = GA(func=lambda x: obj_func(x),
        n_dim=2, size_pop=60, max_iter=120, lb=[-1, -1], ub=[1, 1], precision=[0.00000001, 0.0001])
# 其中lb和up代表的是取值范围
best_x, best_y = ga.run()

print(best_y, best_x)

x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
x, y = np.meshgrid(x, y)
z = np.array(x ** 2 + y ** 2)

fig = plt.figure()
axe = fig.add_subplot(projection="3d")
axe.plot_surface(x, y, z)
axe.scatter3D(best_x[0],best_x[1],best_y,c='r')
axe.set_xlabel('x')
axe.set_ylabel('x')
axe.set_zlabel('x')
axe.set_title("3d")
plt.show()
