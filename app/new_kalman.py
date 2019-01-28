import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints
from filterpy.common import Saver


class MovementModel:
    def __init__(self, initial_position_xy, velocity_xy, Ts, type="constant"):
        self.position_xy = initial_position_xy
        self.initial_position_xy = initial_position_xy
        self.velocity_xy = np.array(velocity_xy)
        self.Ts = Ts
        self.t = 0
        self.update_position()

    def update(self):
        self.t += self.Ts
        self.update_position()

    def update_position(self):
        self.position_xy = np.array(self.initial_position_xy) + self.velocity_xy * self.t

    def get_position(self):
        return self.position_xy


class LocateModel:
    def __init__(self, speaker_positions, movement_model, t_offset=0, C=343):
        self.C = C
        self.speaker_positions = speaker_positions
        self.movement_model = movement_model
        self.t_offset = t_offset

    def update(self):
        self.movement_model.update()

    def get_state(self):
        x, y = self.movement_model.get_position()
        t_offset = self.t_offset
        state = np.array([x, y, t_offset])
        return state

    def get_time(self):
        return self.movement_model.t

    def measurement(self):
        x, y, t_offset = self.get_state()

        dx = self.speaker_positions[:, 0] - x
        dy = self.speaker_positions[:, 1] - y
        d = np.sqrt(dx ** 2 + dy ** 2)
        t = d / self.C
        return t + t_offset

    def noisy_measurement(self, points_t):
        noise = np.random.randn(len(self.speaker_positions)) * points_t
        return self.measurement() + noise

# function that returns the state x transformed by the state transistion function.
# dt is the time step in seconds.
def f_fx(x, dt):
    state_x, velocity_x, state_y, velocity_y, t0 = x
    state_x += velocity_x * dt
    state_y += velocity_y * dt
    return [state_x, velocity_x, state_y, velocity_y, t0]

# Measurement function.
# Converts state vector x into a measurement vector of shape (dim_z).
def f_hx(x):
    state_x, velocity_x, state_y, velocity_y, t0 = x

    def f(speaker):
        return float(np.sqrt((state_x - speaker[0]) ** 2 + (state_y - speaker[1]) ** 2)) / C + t0
    # every speaker sqrt((x-x1)**2+(y-y1)**2) / 343 + t0
    result = [f(speaker) for speaker in speaker_positions]
    return result

def simulate(points_t, value_count=100, plotting=False, initial_position_xy=(0, 0), velocity_xy=(1, 1)):
    # initialize Movement
    movement_model = MovementModel(initial_position_xy=initial_position_xy, Ts=Ts, velocity_xy=velocity_xy)
    # create model object for audio_locate()
    loc_model = LocateModel(speaker_positions=speaker_positions, movement_model=movement_model, t_offset=0.1)

    # define ukf
    ukf = UnscentedKalmanFilter(dim_x=5, dim_z=channels, dt=Ts, hx=f_hx, fx=f_fx, points=points)
    # covariance estimate matrix
    #ukf.P *= 10
    # measurement noise matrix
    ukf.R *= 1e-3
    Q = np.zeros((5, 5))
    Q[:4, :4] = Q_discrete_white_noise(2, Ts, block_size=2, var=10)
    ukf.Q = Q
    saver = Saver(ukf)

    # generate Data
    # simulate n values
    values = value_count
    t = list()
    ground_truth = list()
    for n in range(values):
        ukf_data = loc_model.noisy_measurement(points_t=points_t)
        #ukf_data = model.measurement()
        loc_model.update()
        ukf.predict()
        ukf.update(ukf_data)
        t.append(loc_model.get_time())
        saver.save()
        ground_truth.append(loc_model.get_state())

    ground_truth = np.array(ground_truth)
    t = np.array(t)
    saver.to_array()
    if plotting:
        plt.figure(figsize=(10, 5))
        plt.plot(saver.x[:, 0], saver.x[:, 2], label='ukf')
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], label='track')
        plt.grid()
        plt.legend()
        plt.show()
    else:
        pass


Ts = 0.01
C = 343
points = MerweScaledSigmaPoints(n=5, alpha=.1, beta=2., kappa=1.)

speaker_positions = np.array([[0, 0], [0, 8], [4, 0], [4, 8]])
channels = len(speaker_positions)
simulate(points_t=1e-2, value_count=500, plotting=True, initial_position_xy=(0, 0), velocity_xy=(1, 2))
