import time,sys

from scipy import signal
import numpy as np
import random
#import sympy as sp
import sounddevice as sd
import queue,threading

import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise
from filterpy.kalman import MerweScaledSigmaPoints, JulierSigmaPoints
from filterpy.common import Saver

class AudioLocate:
    __input_channels: int
    __channels: int
    __samples: int
    __duration: int

    def __init__(self, channels: int = 4, samplerate: int = 44100, duration: int = 1, initial_position_xy=(0,0), Ts=0.01) -> object:
        self.__dtimes = list()
        self.initial_position_xy = np.array(initial_position_xy, dtype=float)
        self.Ts = Ts
        self.t = 0
        self.C = 343
        self.__chunks = 1
        self.__output_stream = []
        self.__input_stream = []
        self.__blocksize = 2048
        self.__latency = 1
        self.__channels = channels  # amount of speakers
        self.__duration = duration # duration of noise playback
        self.__samplerate = samplerate # samples - simulate listening for 1 seconds at 44100KHz sample rate
        self.__samples = samplerate * self.__duration
        self.__sources = self.generate_source(self.__channels) # generate n noise channels
        self.__mixed = self.mix()
        self.__recorded = None
        self.__correlation = None
        self.__output_device = None
        self.__input_channels = 1
        self.__input_device = None


    ############# Generate noises ###############

    def generate_source(self, amount: int = 4) -> list:
        sources = []
        for i in range(amount):
            patched = np.tile(np.random.randn(int(self.__samplerate/self.__chunks)),self.__duration*self.__chunks)
            sources.append(patched)
        return sources

    def mix(self):
        sources = iter(self.__sources)
        mix = self.__sources[0]
        next(sources)
        for source in sources:
            mix = np.vstack((mix, source)) # each array shape represents one channel => 4 channels = 4,1 array-shape
            # mix.append(source)
        return mix

    def mix_hdmi(self) -> object:
        """
        MP3/WAV/FLAC datastream
            0		Front Left
            1		Front Right
            2		Center
            3		Subwoofer Freque
            4		Rear Left
            5		Rear Right
            6		Alternative Rear Left
            7		Alternative Rear Right

        DTS/AAC
            1	Front Left
            2	Front Right
            0	Center
            5	Subwoofer Freque
            3	Rear Left
            4	Rear Right
            6	Alternative Rear Left
            7	Alternative Rear Right
        :return:
        """
        if self.__channels == 4:
            for i in range(self.__channels):
                mix = self.__sources[0]
                mix = np.vstack((mix, self.__sources[1]))
                mix = np.vstack((mix, np.zeros(self.__samples)))
                mix = np.vstack((mix, np.zeros(self.__samples)))
                mix = np.vstack((mix, self.__sources[2]))
                mix = np.vstack((mix, self.__sources[3]))
                mix = np.vstack((mix, np.zeros(self.__samples)))
                mix = np.vstack((mix, np.zeros(self.__samples)))
        elif self.__channels == 6:
            for i in range(self.__channels):
                mix = self.__sources[0]
                mix = np.vstack((mix, self.__sources[1]))
                mix = np.vstack((mix, np.zeros(self.__samples)))
                mix = np.vstack((mix, np.zeros(self.__samples)))
                mix = np.vstack((mix, self.__sources[2]))
                mix = np.vstack((mix, self.__sources[3]))
                mix = np.vstack((mix, self.__sources[4]))
                mix = np.vstack((mix, self.__sources[5]))
        else:
            raise ValueError('For HDMI we need 4 or 6 channels! ' + str(self.__channels) + ' are given.')
        self.__mixed = mix

    def mix_mono(self) -> list:
        mix = np.zeros(self.__samples)
        for i in self.__sources:
            mix += 1 / self.__channels * i
        self.__mixed = mix

    ############# Do the math ###############

    def normalize(self) -> None:
        from sklearn.preprocessing import normalize
        self.__recorded = normalize(self.__recorded, axis=0, norm='max')

    def __auto_cross_correlate(self) -> None:
        corr = []
        for source in self.__sources:
            # Pegelcheck um die "besten" Schallquellen zu finden

            # crossCorrelation von original mit umgedrehten recorded
            for i in range(self.__input_channels):
                try:
                    #print("rec",self.__recorded.T[i])
                    corr.append(signal.fftconvolve(source, self.__input_stream[-1][::-1].T[i], mode='same'))
                    #corr.append(signal.fftconvolve(source, self.__recorded.T[i], mode='same'))
                except ValueError as err:
                    print(err)
        self.__correlation = corr

    def __show(self) -> None:
        import matplotlib.pyplot as plt
        try:
            figure, (ax_mixed, *source_plots) = plt.subplots(self.__channels * self.__input_channels + 1, 1)
            ax_mixed.set_title('Recorded noise')
            ax_mixed.plot(self.__recorded)
            counter = 0
        except:
            print("No recording found!")

        try:
            for i in range(self.__channels * self.__input_channels):
                counter += 1
                source_plots[i].set_title('Cross correlation for source channel ' + str(counter))
                color = 'r'
                source_plots[i].plot(np.arange(-len(self.__correlation[i]) + self.__samples,
                                               len(self.__correlation[i])), self.__correlation[i], color)
            figure.tight_layout()
            figure.show()
        except:
            print("No correlation data found to run show()! Please run auto_cross_correlate() before.")


    def __calculate(self,show: bool = False, t0: int = None) -> None:
        self.__auto_cross_correlate()
        self.__location_values = []
        delay_index = []
        delay = []
        try:
            if t0 != None:
                for i in range(len(self.__correlation)):
                    value = (self.__samplerate / 2 - np.argmax(self.__correlation[i])) / self.__samplerate
                    delay_index.append((i, value))
                    delay.append(value)
                if show:
                    for i in delay:
                        print(str(i / self.__duration) + "s")
                        self.__distances = [x / self.__duration for x in delay]
                else:
                    self.__distances = [x / self.__duration for x in delay]
            else:
                # t0 is set to the speaker, who gets a signal first
                for i in range(len(self.__correlation)):
                    delay_index.append((i, np.argmax(self.__correlation[i])))
                    delay.append(np.argmax(self.__correlation[i]))
                    min_delay = max(delay)
                    delays = []
                    for j in delay:
                        delays.append(min_delay - j)
                #for i in delays:
                self.__distances = [x / self.__duration for x in delays]
                self.__dtimes = [x / self.__samplerate for x in delays]
                print("dtimes", self.__dtimes)
        except TypeError:
            print("No values to calculate()! Please run auto_cross_correlate() before.")

    ############# Playback and recording ###############

    def get_devices(self):
        return sd.query_devices()

    def playrec(self) -> None:
        try:
            self.__t_start = int(time.time())
            self.__recorded = sd.playrec(self.__mixed.T,
                                         self.__samplerate,
                                         device=(self.__input_device, self.__output_device),
                                         channels=self.__input_channels)
            self.__t_stop = int(time.time())
            self.__t_delta = int(time.time()) - self.__t_start
        except (sd.PortAudioError) as err:
            print("Something went wrong!", err, "- Check your output settings and set_output_device()")
            print("\nAvailable devices:", self.get_devices())
            print(self.__mixed.shape)
        sd.wait()

    def start_audio_analyses(self):
        self._play_thread()
        self._rec_thread()

    def _rec_thread(self):
        try:
            q = queue.Queue()

            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                if status:
                    print(status)
                q.put(indata.copy())

            with sd.InputStream(samplerate=self.__samplerate,blocksize=self.__samplerate, device=self.__input_device,
                                channels=self.__input_channels, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the calculating')
                print('#' * 80)
                N = 100
                for _ in range(N):
                    self.__input_stream.append(q.get())
                    self.__calculate()
                    self.kalcalc()
                self.print_track()
                #while True:
                #    print("\nCalculating...")
                #    self.__input_stream.append(q.get())
                #    self.__calculate()
                #    self.kalcalc()



        except KeyboardInterrupt:
            sd.stop()
            print('\nRecording finished: ')

    def _play_thread(self):
        try:
            """
            Playing the mixed noise in a loop
            """
            try:
                print("Playing loop..")
                sd.play(self.__mixed.T, self.__samplerate, device=self.__output_device, loop=True)
            except (sd.PortAudioError) as err:
                print("Something went wrong!", err, "- Check your output settings and set_output_device()")
                print("\nAvailable devices:", self.get_devices())
                print(self.__mixed.shape)
        except KeyboardInterrupt:
            print('\nPlaying finished... ')

    ##### DEBUG STUFF

    def play_mono_mix(self) -> None:
        """
        Playing the mixed noise for debug purpose
        """
        sd.play(self.__mixed, self.__samples, blocking=True)

    def play_mix(self) -> None:
        """
        Playing the mixed noise for debug purpose
        """
        try:
            sd.play(self.__mixed.T, self.__samplerate, device=self.__output_device, blocking=True)  # NEEDED TO TRANSFORM
        except (sd.PortAudioError) as err:
            print("Something went wrong!", err, "- Check your output settings and set_output_device()")
            print("\nAvailable devices:", self.get_devices())

    def play_sources(self) -> None:
        """
        Playing the source noise for debug purpose
        """
        for source in self.__sources:
            sd.play(source, self.__samplerate, blocking=True)

    def play_recorded(self) -> None:
        """
        Playing the source noise for debug purpose
        """
        for source in self.__recorded.T:
            sd.play(source, self.__samplerate, blocking=True)

    ##### SETTER

    def set_output_device(self, output_device: int) -> None:
        """
        Set the output device
        Get a list from get_devices()
        :int output_device:
        :return:
        """
        self.__output_device = output_device

    def set_input_device(self, input_device: int) -> None:
        """
        Set the output device
        Get a list from get_devices()
        :int output_device:
        :return:
        """
        self.__input_device = input_device

    def set_duration(self, duration: int) -> None:
        """
        Set the duration for recording in seconds
        :param duration:
        """
        self.__duration = duration
        self.__samples = self.__samplerate*self.__duration

    def get_duration(self) -> int:
        """
        get the duration of recordings in seconds
        :return:
        """
        return self.__duration

    def get_rec_duration(self) -> int:
        """
        get the duration of recordings in seconds
        :return:
        """
        return self.__samples/self.__samplerate

    def set_speaker_locations(self,*positions: tuple) -> None:
        if len(positions) != self.__channels:
            raise ValueError("Got "+str(len(positions))+" positions but there are "+str(self.__channels)+" speakers.")
        else:
            positionList = []
            for pos in positions:
                if type(pos) != tuple:
                    raise ValueError("Need tuples for positions - eg. (x1,y1),(x2,y2). "
                                     "First value should be (0,0)!")
                else:
                    positionList.append(pos)
            self.__speaker_locations = np.array(positionList)

    def set_position(self,t0=1):
        self.__calculate(t0)

    ##### GETTER

    def get_distances(self) -> list:
        try:
            return self.__distances
        except:
            AttributeError("No distances to return. Run calculate().")

    def get_dtimes(self) -> list:
        try:
            self.__calculate()
            return self.__dtimes
        except:
            AttributeError("No distances to return. Run calculate().")

    def get_position(self) -> list:
        try:
            return self.__position
        except:
            AttributeError("No distances to return. Run calculate().")

    def get_input_stream(self) -> list:
        try:
            return self.__input_stream
        except:
            AttributeError("No inputStream to return. Run calculate().")



    ######

    # function that returns the state x transformed by the state transistion function.
    # dt is the time step in seconds.
    def update(self):
        self.t += self.Ts
        self.update_position()

    def update_position(self):
        self.set_position(self.t)

    def get_position(self):
        return self.get_position()

    def get_time(self):
        return self.t

    def measurement(self):
        t_offset = 0.1
        t = np.array(self.get_dtimes())
        return t + t_offset

    def init_kalman(self, plotting=False):
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
                return float(np.sqrt((state_x - speaker[0]) ** 2 + (state_y - speaker[1]) ** 2)) / 343 + t0

            # every speaker sqrt((x-x1)**2+(y-y1)**2) / 343 + t0
            result = [f(speaker) for speaker in self.__speaker_locations]
            return result

        points = MerweScaledSigmaPoints(n=5, alpha=.1, beta=2., kappa=1.)
        # points = JulierSigmaPoints(n=5, kappa=1)

        # define ukf
        self.ukf = UnscentedKalmanFilter(dim_x=5, dim_z=self.__channels, dt=self.Ts, hx=f_hx, fx=f_fx, points=points)
        # covariance estimate matrix
        self.ukf.P *= 10
        # measurement noise matrix
        self.ukf.R *= 1e-3
        Q = np.zeros((5, 5))
        Q[:4, :4] = Q_discrete_white_noise(2, self.Ts, block_size=2, var=10)
        self.ukf.Q = Q
        self.saver = Saver(self.ukf)
        self.t_list = list()



    def print_track(self):
        t = np.array(self.t_list)
        self.saver.to_array()
        saver = self.saver
        plt.figure(figsize=(10, 5))
        plt.plot(saver.x[:, 0], saver.x[:, 2], label='ukf')
        for idx, s in enumerate(self.__speaker_locations):
            color = 'rgby'
            plt.plot(s[0], s[1], marker='s', ms=10, color=color[idx])
        plt.grid()
        plt.legend()
        plt.show()

        plt.figure(figsize=(10, 5))
        plt.plot(t, saver.x[:, 1], label='velocity_x')
        plt.plot(t, saver.x[:, 3], label='velocity_y')
        plt.grid()
        plt.legend()
        plt.show()


    def kalcalc(self):
        ukf_data = self.measurement()
        print("ukf", ukf_data)
        self.update()
        self.ukf.predict()
        self.ukf.update(ukf_data)
        self.t_list.append(self.get_time())
        self.saver.save()

if __name__ == "__main__":
    test = AudioLocate(4, samples=44100)
