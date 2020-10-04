import numpy as np
from matplotlib import pyplot as plt
from imageanalysistools.rita import rita
from imageanalysistools.rital import rital

## Generate a few points

N = 100
x = np.random.randn(2,N)
th = 0.1
R = np.array([[np.cos(th), np.sin(th)],[-np.sin(th), np.cos(th)]])
x = R@np.array([[10, 0],[0, 100]])@x
T = 10
x = x + np.tile(np.array([[T],[0]]),(1,N))
xh = np.append(x, np.ones((1,N)), 0)

plt.figure(1)
plt.clf()
rita(x,'*')
plt.axis([T-100, T+100, -200, 200])

## Ordinary least squares

# y = p1(1)*x + p1(2);
# l = [p1(1) -1 p1(2)]
#
# Algorithm 1
#
xx = x[0,:]
yy = x[1,:]
ee = xh[2,:]
p1, *_ = np.linalg.lstsq(np.array([xx, ee]).T, yy, rcond=None) # least squares
l1 = np.array([p1[0], -1, p1[1]])
#
# Normalization
l1n = l1/np.linalg.norm(l1[0:2])

# plot
plt.figure(2)
plt.clf()
rita(x,'*')
plt.axis([T-100, T+100, -200, 200])
rital(l1n,'r')

# calculate error
e1 = np.sqrt(np.sum( (l1n@xh)**2 ))

## Total least squares

# Algorithm 2
#
mm = np.mean(x.T, 0)
cc = np.cov(x)
u,s,vh=np.linalg.svd(cc)
rr = u[:,0]
l2 = np.cross(np.append(mm.T,1),np.append(rr,0))
#
# Normalization
l2n = l2/np.linalg.norm(l2[0:2])


# plot
plt.figure(3)
plt.clf()
rita(x,'*')
plt.axis([T-100, T+100, -200, 200])
rital(l1n,'r')
rital(l2n,'g')

# calculate error
e2 = np.sqrt(np.sum( (l2n@xh)**2 ))

## SVD hacket

# Algorithm 3
#
u,s,vh=np.linalg.svd(xh.T)
l3 = vh[2]
#
# Normalization
l3n = l3/np.linalg.norm(l3[0:2])

# plot
plt.figure(4)
plt.clf()
rita(x,'*')
plt.axis([T-100, T+100, -200, 200])
rital(l1n,'r')
rital(l2n,'g')
rital(l3n,'k')

# calculate error
e3 = np.sqrt(np.sum( (l3n@xh)**2 ))

## Summary

print('Errors measured in orthonogal direction. Tried three algorithms')
print('Error using algorithm 1: ' + str(e1) + ' minimizing least squares (error in y-direction)')
print('Error using algorithm 2: ' + str(e2) + ' minimizing total least squares (error in orthogonal direction)')
print('Error using algorithm 3: ' + str(e3) + ' using svd-trick')

plt.show()
