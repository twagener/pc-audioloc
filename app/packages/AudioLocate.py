from scipy import signal
import numpy as np
import sounddevice as sd


class AudioLocate:
    amount: int
    samples: int
    rec_duration: int

    def __init__(self, amount: int = 4, samples: int = 44100) -> object:
        self.amount = amount  # amount of speakers
        self.samples = samples  # samples - simulate listening for 1 seconds at 44100KHz sample rate
        self.sources = self.generate_source(self.amount)
        self.mixed = self.mix()
        self.rec_duration = 2  # seconds (2 times the sample)
        self.recorded = self.fake_input()
        self.__correlation = None

    def generate_source(self, amount: int = 4) -> list:
        sources = []
        for i in range(amount):
            sources.append(np.random.randn(self.samples))
        return sources

    def mix(self):
        sources = iter(self.sources)
        mix = self.sources[0]
        next(sources)
        for source in sources:
            mix = np.vstack((mix, source))
            # mix.append(source)
        return mix

    def fake_input(self) -> list:
        mix = np.zeros(self.samples)
        for i in self.sources:
            mix += 1 / self.amount * i
        return mix

    def fake_input_with_failure(self) -> None:
        import random
        rand_fail = random.randint(0, self.amount - 1)
        counter = 0
        mix = np.zeros(self.samples)
        for i in self.sources:
            if counter == rand_fail:
                mix += 1 / self.amount * self.generate_source(1)[0]
            else:
                mix += 1 / self.amount * i
            counter += 1
        self.recorded = mix

    def fake_input_shift(self, lower: int = 10, upper: int = 500) -> None:
        import random
        mix = np.zeros(self.samples)
        for source in self.sources:
            shift = random.randint(-upper, -lower)
            mix += 1 / self.amount * np.roll(source, shift)
        self.recorded = mix

    def fake_input_shift_spec(self, shift: int = -441, spec: int = 1) -> None:
        """
        :rtype: None
        """
        # shift - shift random speaker by -441 samples (~3,43m) ?
        counter = 1
        mix = np.zeros(self.samples)
        for source in self.sources:
            if counter != spec:
                mix += 1 / self.amount * source
            else:
                mix += 1 / self.amount * np.roll(source, shift)
            counter += 1
        self.recorded = mix

    def auto_cross_correlate(self) -> None:
        corr = []
        for source in self.sources:
            # Pegelcheck um die "besten" Schallquellen zu finden

            # crossCorrelation von original mit umgedrehten recorded
            corr.append(signal.fftconvolve(source, self.recorded[::-1], mode='same'))
        self.__correlation = corr

    def show(self) -> None:
        import matplotlib.pyplot as plt
        figure, (ax_mixed, *source_plots) = plt.subplots(self.amount + 1, 1)
        ax_mixed.set_title('White noise')
        ax_mixed.plot(self.recorded)
        counter = 0
        for i in range(self.amount):
            counter += 1
            source_plots[i].set_title('CrossCorr for ' + str(counter))
            if max(self.__correlation[i]) > self.samples / len(self.__correlation) - max(self.recorded) * 100:
                color = 'g'
            else:
                color = 'r'
            source_plots[i].plot(np.arange(-len(self.__correlation[i]) + self.samples, len(self.__correlation[i])),
                                 self.__correlation[i], color)
        figure.tight_layout()
        figure.show()

    def calculate(self) -> None:
        self.location_values = []
        try:
            for i in range(len(self.__correlation)):
                delay_index = np.argmax(self.__correlation[i])
                norm_samples = self.samples / 2
                delta = (delay_index - norm_samples) / self.samples
                distance = delta * 343
                self.location_values.append((delta, distance))
        except TypeError:
            print("No values to calculate! Please run auto_cross_correlate() before.")

    def print_locations(self):
        try:
            for loc in self.location_values:
                print("Delta t:" + str(loc[0]) + "ms" + " und ist " + str(loc[1]) + "m entfernt.")
        except AttributeError:
            print("No values found for locations! Run calculate() first.")

    def get_devices(self):
        return sd.query_devices()

    def play_mono_mix(self) -> None:
        """
        Playing the mixed noise for debug purpose
        """
        sd.play(self.mixed, self.samples, blocking=True)

    def play_mix(self) -> None:
        """
        Playing the mixed noise for debug purpose
        """
        try:
            sd.play(self.mixed.T, self.samples, device=2, blocking=True) # NEEDED TO TRANSFORM
        except (sd.PortAudioError) as err:
            print("something went wrong!",err," Check your output settings.")
            print("\n",self.get_devices())
            #print(sd.default.device)
            #print(sd.check_output_settings(device=2))
            print(self.mixed.shape)

    def play_sources(self) -> None:
        """
        Playing the source noise for debug purpose
        """
        for source in self.sources:
            sd.play(source, self.samples, blocking=True)

    def set_rec_duration(self, duration: int) -> None:
        """
        Set the duration for recording in seconds
        :param duration:
        """
        self.rec_duration = duration

    def get_rec_duration(self) -> int:
        """
        get the duration of recordings in seconds
        :return:
        """
        return self.rec_duration

    def record(self) -> None:
        duration = self.rec_duration
        self.myrecording = sd.rec(duration * self.samples, samplerate=self.samples, channels=1)


if __name__ == "__main__":
    test = AudioLocate(4, samples=44100)
