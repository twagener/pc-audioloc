import time

from scipy import signal
import numpy as np
import sounddevice as sd
import queue


class AudioLocate:
    __input_channels: int
    __channels: int
    __samples: int
    __duration: int

    q = queue.Queue()

    def __init__(self, channels: int = 4, samplerate: int = 44100, duration: int = 1) -> object:
        self.__channels = channels  # amount of speakers
        self.__duration = duration
        self.__samplerate = samplerate # samples - simulate listening for 1 seconds at 44100KHz sample rate
        self.__samples = samplerate * self.__duration
        self.__sources = self.generate_source(self.__channels)
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
            patched = np.tile(np.random.randn(self.__samplerate),self.__duration)
            sources.append(patched)
        return sources

    def mix(self):
        sources = iter(self.__sources)
        mix = self.__sources[0]
        next(sources)
        for source in sources:
            mix = np.vstack((mix, source))
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

    def fake_input(self) -> list:
        mix = np.zeros(self.__samples)
        for i in self.__sources:
            mix += 1 / self.__channels * i
        return mix

    def fake_input_with_failure(self) -> None:
        import random
        rand_fail = random.randint(0, self.__channels - 1)
        counter = 0
        mix = np.zeros(self.__samples)
        for i in self.__sources:
            if counter == rand_fail:
                mix += 1 / self.__channels * self.generate_source(1)[0]
            else:
                mix += 1 / self.__channels * i
            counter += 1
        self.__recorded = mix.T

    def fake_input_shift(self, lower: int = 10, upper: int = 500) -> None:
        import random
        mix = np.zeros(self.__samples)
        for source in self.__sources:
            shift = random.randint(-upper, -lower)
            mix += 1 / self.__channels * np.roll(source, shift)
        self.__recorded = mix.T

    def fake_input_shift_spec(self, shift: int = -441, spec: int = 1) -> None:
        """
        :rtype: None
        """
        # shift - shift random speaker by -441 samples (~3,43m) ?
        counter = 1
        mix = np.zeros(self.__samples)
        for source in self.__sources:
            if counter != spec:
                mix += 1 / self.__channels * source
            else:
                mix += 1 / self.__channels * np.roll(source, shift)
            counter += 1
        self.__recorded = mix

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
        figure, (ax_mixed, *source_plots) = plt.subplots(self.__channels * self.__input_channels + 1, 1)
        ax_mixed.set_title('Recorded noise')
        ax_mixed.plot(self.__recorded)
        counter = 0
        for i in range(self.__channels * self.__input_channels):
            counter += 1
            source_plots[i].set_title('Cross correlation for source number ' + str(counter))
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
        except TypeError:
            print("No values to calculate! Please run auto_cross_correlate() before.")

        minDelay = max(delay)
        delays = []
        for i in delay:
            delays.append(minDelay-i)

        for i in delays:
            #print(str(((i/self.__samplerate)+system_delta)*343)+"m")
            print(str(((i / self.__samplerate)) * 343) + "m")

    def print_locations(self):
        try:
            for loc in self.location_values:
                print("Delta t:" + str(loc[0]) + "ms" + " und ist " + str(loc[1]) + "m entfernt.")
        except AttributeError:
            print("No values found for locations! Run calculate() first.")

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



if __name__ == "__main__":
    test = AudioLocate(4, samples=44100)
