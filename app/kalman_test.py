

import filterpy as filterpy
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import unscented_transform, MerweScaledSigmaPoints
from filterpy.kalman import UnscentedKalmanFilter as UKF

import math

from numpy.linalg import norm
import random
import numpy as np



# für x in der Form (x,y,t0)
#und dt als Delta t

def f(x,dt):
    F = np.array([[1,0,0],
                  [0,1,0],
                  [0,0,1]],dtype=float)
    return np.dot(F,x)

def h(x):
    t1 = distance(h.speaker_1,x);
    t2 = distance(h.speaker_2,x);
    t3 = distance(h.speaker_3,x);
    t4 = distance(h.speaker_4,x);
    return [t1,t2,t3,t4]

def distance(speaker, x):
    dx = math.fabs(x[0] - speaker[0])
    dy = math.fabs(x[1] - speaker[1])
    dist = math.sqrt(dx**2 + dy**2)
    dist = (dist/343) + x[2]
    return dist

class SoundMic(object):

    def __init__(self, speaker_1_pos, speaker_2_pos, speaker_3_pos, speaker_4_pos, t1_std, t2_std, t3_std, t4_std):
        self.speaker_1_pos = np.asarray(speaker_1_pos)
        self.speaker_2_pos = np.asarray(speaker_2_pos)
        self.speaker_3_pos = np.asarray(speaker_3_pos)
        self.speaker_4_pos = np.asarray(speaker_4_pos)
        self.t1_std = t1_std
        self.t2_std = t2_std
        self.t3_std = t3_std
        self.t4_std = t4_std

    def reading_of(self, mic_pos):
        t1 = distance(self.speaker_1_pos, x);
        t2 = distance(self.speaker_2_pos, x);
        t3 = distance(self.speaker_3_pos, x);
        t4 = distance(self.speaker_4_pos, x);

        return t1, t2, t3, t4

    def noisy_reading(self, mic_pos):
        # t1_noise, t2_noise, t3_noise, t4_noise = self.reading_of(mic_pos)
        t1_noise = distance(self.speaker_1_pos, mic_pos)
        t2_noise = distance(self.speaker_2_pos, mic_pos)
        t3_noise = distance(self.speaker_3_pos, mic_pos)
        t4_noise = distance(self.speaker_4_pos, mic_pos)

        # t1_noise += random.uniform(0, 0.1) * self.t1_std
        # t2_noise += random.uniform(0, 0.1) * self.t2_std
        # t3_noise += random.uniform(0, 0.1) * self.t3_std
        # t4_noise += random.uniform(0, 0.1) * self.t4_std

        return t1_noise, t2_noise, t3_noise, t4_noise


class MICSim(object):
    def __init__(self, x, y, t0):
        self.x = x
        self.y = y
        self.t0 = t0

    def update(self):
        return self.x, self.y, self.t0

x = 3.  # meters
y = 2.
t0 = 0.
dt = 1.

h.speaker_1 = (0., 0.)
h.speaker_2 = (0., 8.)
h.speaker_3 = (4., 8.)
h.speaker_4 = (4., 0.)

# UKF(Dimension Zustand, Dimension Messung, Zeit zwischen Schritten, f, h, Klasse der Sigma Punkte)
points = MerweScaledSigmaPoints(n=3, alpha=.1, beta=2., kappa=0.)
kf = UKF(3, 4, 1, fx=f, hx=h, points=points)


ukf = UnscentedKalmanFilter(dim_x=5, dim_z=NSpeaker, dt=Ts, hx=f_C, fx=f_Ad, points=sigmas)
ukf.P *= 100
ukf.R *= 1e-4
Q = np.zeros( (5, 5) )
Q[:4, :4] = Q_discrete_white_noise(2, Ts, block_size=2, var=10)
ukf.Q = Q
saver = Saver(ukf)

# kf.Q[0:2, 0:2] = Q_discrete_white_noise(2, dt=dt, var=0)
Ts = 1
Gd = np.matrix([[Ts, 0],
                [Ts, 0],
                [0, Ts]])
Q = np.diag((0.1, 0.1))
kf.Q = Gd * Q * Gd.T

# kf.Q[2,2] = 0

kf.R = np.diag([x ** 2, y ** 2, t0 ** 2, 1 ** 2])
kf.x = np.array([3, 2, 0])
kf.P = np.diag([0.3 ** 2, 0.3 ** 2, 15 ** 2])

np.random.seed(200)
pos = (0, 0)
mic = SoundMic((0, 0), (0, 8), (4, 8), (4, 0), 0.0105, 0.195, 0.0177, 0.0065)
ac = MICSim(3, 2, 0)

time = np.arange(0, 360 + dt, dt)
xs = []
for _ in range(1, 100):
    ac.update()
    # print([x,y,t0])
    # print(mic.speaker_1_pos)
    # print(distance(mic.speaker_1_pos,[x,y,t0]))
    r = mic.noisy_reading([x, y, t0])

    kf.predict()

    #print(80 * "#" + "\n", kf)

    # print("r: ",r)
    kf.update([r[0], r[1], r[2], r[3]])

    xs.append(kf.x)

