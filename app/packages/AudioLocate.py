from scipy import signal
import numpy as np
import sounddevice as sd


class AudioLocate:
    amount: int
    samples: int
    recDuration: int

    def __init__(self,amount:int = 4,samples: int = 44100) -> object:
        self.amount = amount #amount of speakers
        self.samples = samples #samples - simulate listening for 1 seconds at 44100KHz sample rate
        self.sources = self.generate_source(self.amount)
        self.mixed = self.mix()
        self.recDuration = 2 # seconds (2 times the sample)

    def generate_source(self, amount: int = 4) -> list:
        sources = []
        for i in range(amount):
            sources.append(np.random.randn(self.samples))
        return sources

    def mix(self) -> list:
        mix = np.zeros(self.samples)
        for i in self.sources:
            mix += 1 / self.amount * i
        return mix

    def mix_with_failure(self) -> None:
        import random
        rand_fail = random.randint(0,self.amount-1)
        counter = 0
        mix = np.zeros(self.samples)
        for i in self.sources:
            if counter == rand_fail:
                mix += 1 / self.amount * self.generate_source(1)[0]
            else:
                mix += 1 / self.amount * i
            counter += 1
        self.mixed = mix

    def mix_shift(self, lower: int = 10, upper: int = 500) -> None:
        import random
        mix = np.zeros(self.samples)
        for i in self.sources:
            shift = random.randint(-upper, -lower)
            mix += 1 / self.amount * np.roll(i,shift)
        self.mixed = mix

    def mix_shift_spec(self, shift: int = -441, spec: int = 1) -> None:
        """
        :rtype: None
        """
        # shift - shift random speaker by -441 samples (~3,43m) ?
        counter = 1
        mix = np.zeros(self.samples)
        for i in self.sources:
            if counter != spec:
                mix += 1 / self.amount * i
            else:
                mix += 1 / self.amount * np.roll(i,shift)
            counter += 1
        self.mixed = mix

    def auto_cross_correlate(self) -> None:
        recorded = self.mixed
        corr = []
        for source in self.sources:
            # Pegelcheck um die "besten" Schallquellen zu finden

            # crossCorrelation von original mit umgedrehten recorded
            corr.append(signal.fftconvolve(source, recorded[::-1], mode='same'))
        self.corr = corr

    def show(self):
        import matplotlib.pyplot as plt
        figure, (ax_mixed, *sourcePlots) = plt.subplots(self.amount + 1, 1)
        ax_mixed.set_title('White noise')
        ax_mixed.plot(self.mixed)
        counter = 0
        for i in range(self.amount):
            counter += 1
            sourcePlots[i].set_title('CrossCorr for ' + str(counter))
            if max(self.corr[i]) > self.samples/len(self.corr)-max(self.mixed)*100:
                color = 'g'
            else:
                color = 'r'
            sourcePlots[i].plot(np.arange(-len(self.corr[i]) + self.samples, len(self.corr[i])), self.corr[i], color)
        figure.tight_layout()
        figure.show()

    def calculate(self):
        self.locationValues = []
        for i in range(len(self.corr)):
            delayIndex = np.argmax(self.corr[i])
            normSamples=self.samples/2
            delta = (delayIndex-normSamples)/self.samples
            distance = delta*343
            self.locationValues.append((delta,distance))

    def print_locations(self):
        for loc in self.locationValues:
            print("Delta t:" + str(loc[0]) + "ms" + " und ist " + str(loc[1]) + "m entfernt.")

    def play_mix(self) -> None:
        sd.play(self.mixed, self.samples, blocking=True)

    def play_sources(self) -> None:
        for source in self.sources:
            sd.play(source, self.samples, blocking=True)

    def set_rec_duration(self, duration: int) -> None:
        self.recDuration = duration

    def get_rec_duration(self) -> int:
        return self.recDuration

    def record(self) -> None:
        duration = self.recDuration
        self.myrecording = sd.rec(duration * self.samples, samplerate=self.samples, channels=1)


if __name__ == "__main__":
    test = AudioLocate(4,samples=44100)
    test.mix_shift()
    #test.mix_shift_spec(-4410,4) # Set a distance of 34.3m

    #test.mix_with_failiure()
    test.auto_cross_correlate()
    test.show()
    test.calculate()
    test.print_locations()
