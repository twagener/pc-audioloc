import time

from scipy import signal
import numpy as np
#import sympy as sp
import sounddevice as sd
import queue


class AudioLocate:
    __input_channels: int
    __channels: int
    __samples: int
    __duration: int

    def __init__(self, channels: int = 4, samplerate: int = 44100, duration: int = 1) -> object:
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
        self.__speaker_locations = np.array()


    ############# Generate noises ###############

    def generate_source(self, amount: int = 4) -> list:
        sources = []
        for i in range(amount):
            patched = np.tile(np.random.randn(self.__samplerate),self.__duration)
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

    ############# Fake stuff for debugging ###############
    def fake_pos(self,x:int = 0,y: int = 0):
        try:
            xy = [x,y]
            self.__distances = np.sqrt(np.sum((self.__speaker_locations-xy)**2,axis=1))
            return True
        except:
            print("No speaker locations found! Please run set_speaker_locations() before")
            return False

    def fake_recording_pos(self,x: int = 5,y: int = 5) -> bool:
        if x > 0 and y > 0:
            mix = np.zeros(self.__samples)
            for source in self.__sources:
                shift = 10
                mix += 1 / self.__channels * np.roll(source, shift)
            self.__recorded = np.vstack((mix))
            return True
        else:
            return False

    ### FAKE OUTPUT

    @staticmethod
    def fake_output_times(x: int = 0, y: int = 0) -> tuple:
        t0 = 0
        t1 = 1
        t2 = 1
        t3 = 1

        return (t0,t1,t2,t3)


    ############# Do the math ###############

    def normalize(self) -> None:
        from sklearn.preprocessing import normalize
        self.__recorded = normalize(self.__recorded, axis=0, norm='max')

    def auto_cross_correlate(self) -> None:
        corr = []
        for source in self.__sources:
            # Pegelcheck um die "besten" Schallquellen zu finden

            # crossCorrelation von original mit umgedrehten recorded
            for i in range(self.__input_channels):
                try:
                    corr.append(signal.fftconvolve(source, self.__recorded[::-1].T[i], mode='same'))
                except ValueError as err:
                    print(err)
        self.__correlation = corr

    def show(self) -> None:
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
                #source_plots[i].set_xlim([200000,210000])
    #            if max(self.__correlation[i]) > self.__samples / len(self.__correlation) - max(self.__recorded) * 100:
    #                color = 'g'
    #            else:
    #                color = 'r'
                color = 'r'
                source_plots[i].plot(np.arange(-len(self.__correlation[i]) + self.__samples,
                                               len(self.__correlation[i])), self.__correlation[i], color)
            figure.tight_layout()
            figure.show()
        except:
            print("No correlation data found to run show()! Please run auto_cross_correlate() before.")

    def calculate(self) -> None:
        self.location_values = []
        # seconds before the first sound arrives
        system_delta = np.argwhere(self.__recorded > 0.01)[0][0]/self.__samplerate
        delay_index = []
        delay = []

        try:
            for i in range(len(self.__correlation)):
                delay_index.append((i,np.argmax(self.__correlation[i])))
                delay.append(np.argmax(self.__correlation[i]))
                #delay_index_noise = np.argmax(self.__recorded)
                #norm_samples = self.__samples / 2
                #print(self.__t_start)
                #delta = (delay_index - norm_samples) / self.__samples
                #delta = self.__t_start
                #distance = delta * 343
                minDelay = max(delay)
                delays = []
                for i in delay:
                    delays.append(minDelay - i)
            print(delays)
            for i in delays:
                # print(str(((i/self.__samplerate)+system_delta)*343)+"m")
                print(str(((i / self.__samplerate)) * 343) + "m")
        except TypeError:
            print("No values to calculate()! Please run auto_cross_correlate() before.")



    def print_locations(self):
        try:
            from sympy.geometry import Circle,intersection
            for loc in self.location_values:
                print("Delta t:" + str(loc[0]) + "ms" + " und ist " + str(loc[1]) + "m entfernt.")
                c0 = Circle((0,0), loc[0])
                c1 = Circle((0, 0), loc[1])
                c2 = Circle((0, 0), loc[2])
                c3 = Circle((0, 0), loc[3])
                print(intersection(c1,c2,c3))
        except AttributeError:
            print("No values found for print_locations()! Run calculate() first.")

    ############# Playback and recording ###############

    def get_devices(self):
        return sd.query_devices()

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
            self.__play_start = int(time.time())
            print("Playing..")
            sd.play(self.__mixed.T, self.__samplerate, device=self.__output_device)  # NEEDED TO TRANSFORM
            print(". play done.")
        except (sd.PortAudioError) as err:
            print("Something went wrong!", err, "- Check your output settings and set_output_device()")
            print("\nAvailable devices:", self.get_devices())
            print(self.__mixed.shape)

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

    def rec(self) -> None:
        try:
            print("Recording..")
            self.__rec_start = int(time.time())

            self.__recorded = sd.rec(self.__samplerate, device=self.__input_device, channels=self.__input_channels)

            self.__rec_stop = int(time.time())
            self.__rec_delta = int(time.time()) - self.__rec_start
            print(". rec done.")
        except (sd.PortAudioError) as err:
            print("Something went wrong!", err, "- Check your output settings and set_output_device()")
            print("\nAvailable devices:", self.get_devices())
            print(self.__mixed.shape)


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

    def set_speaker_locations(self,x1,y1):
        pass



if __name__ == "__main__":
    test = AudioLocate(4, samples=44100)
