"""
we want to interpolate a value which is beyond the input range
"""
# import os
# path = '/usr/local/bin'
# for entry in os.scandir(path):
#     if not entry.name.startswith('.') and entry.is_file():
#         print(entry.name)

# constant extrapolation, just like fill
from scipy import interp, arange, exp
from scipy import interpolate
x = [1,4,8]
y = [0.2, 0.5, 1.2]
f = interpolate.interp1d( x, y, fill_value='extrapolate')
print(f([0,11]))

## linear extrapolation
from scipy.interpolate import interp1d
from scipy import arange, array, exp
def expera1d(interpolator):
    xs = interpolator.x
    ys = interpolator.y

    def pointwise(x):
        if x < xs[0]:
            return ys[0] + (x-xs[0]) * (ys[1]-ys[0])/ (xs[1] -xs[0])
        elif x > xs[-1]:
            return ys[-1] + (x-xs[-1])*(ys[-1]-ys[-2])/(xs[-1]-xs[-2])
        else:
            return interpolator(x)

    def ufunclike(xs):
        return array(map(pointwise, array(xs)))

    return ufunclike

x = [1,4,8]
y = [0.2, 0.5, 1.2]
f_i = interp1d(x, y)
f_x = expera1d(f_i)
list1 = f_x(arange(10)).tolist()

## use * for unpacking
# * iterable unpacking operator and ** dictionary unpacking operator
print(*list1)

## modified based on this
import numpy as np
def extrap(x,xp, yp):
    y = np.interp(x, xp, yp)
    y[x<xp[0]] = yp[0] + (x[x<xp[0]] - xp[0]) * (yp[0] - yp[1]) /(xp[0] - xp[1])
    y[x>xp[0]] = yp[-1] + (x[x>xp[-1]]-xp[-1]) * (yp[-1]-yp[-2])/(xp[-1] - xp[-2])
    return y

x_a = [1,4,8]
y_a = [0.2, 0.5, 1.2]

xtest = np.array([1,10])
## not work
print(np.interp([1,10],x_a,y_a))
## now work
print(extrap(xtest, x_a, y_a))
## second way
print('\nNew method below \n')

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
import matplotlib.pyplot as plt

xi = np.array([1,4,8,9])
yi = np.array([0.2, 0.5, 1.2,1.5])

# positions to interpolate
x = np.arange(20)
# spline order: 1 linear, 2, quadratic, 3 cubic
order = 1
s = InterpolatedUnivariateSpline(xi, yi, k=order)
y = s(x)

print(y)

plt.figure()
plt.plot(xi, yi)
for order in range(1, 4):
    s = InterpolatedUnivariateSpline(xi, yi, k=order)
    y=s(x)
    plt.plot(x, y, label = order)
# plt.show()

## third way
import scipy
tck = scipy.interpolate.splrep([1, 2, 3, 4, 5], [1, 4, 9, 16, 25], k=1, s=0)
print(scipy.interpolate.splev(1.5, tck))

##



