from scipy import signal
import numpy as np

class AudioLocate:

    def __init__(self,amount=4,samples=44100):
        self.amount = amount #amount of speakers
        self.samples = samples #samples - simulate listening for 2 seconds at 22050KHz sample rate
        self.sources = self.generateSource(self.amount)
        self.mixed = self.mixIt()

    def generateSource(self,amount=4):
        sources = []
        for i in range(amount):
            sources.append(np.random.randn(self.samples))
        return sources

    def mixIt(self):
        mix = np.zeros(self.samples)
        for i in self.sources:
            mix += 1 / self.amount * i
        return mix

    def mixFail(self):
        import random
        randFail = random.randint(0,self.amount-1)
        counter = 0
        mix = np.zeros(self.samples)
        for i in self.sources:
            if counter == randFail:
                mix += 1 / self.amount * self.generateSource(1)[0]
            else:
                mix += 1 / self.amount * i
            counter += 1
        self.mixed = mix

    def mixShift(self,lower=10,upper=500):
        import random
        mix = np.zeros(self.samples)
        for i in self.sources:
            shift = random.randint(-upper, -lower)
            mix += 1 / self.amount * np.roll(i,shift)
        self.mixed = mix

    def mixShiftSpec(self,shift=-441,spec=1):
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

    def autocorr(self):
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

    def showDiff4(self):
        import matplotlib.pyplot as plt
        figure, (ax_mixed, *sourcePlots) = plt.subplots(self.amount + 1, 1)
        ax_mixed.set_title('White noise')
        ax_mixed.plot(self.mixed)
        counter = 0
        for i in range(self.amount):
            counter += 1
            sourcePlots[i].set_title('CrossCorr for ' + str(counter))
            if counter == 1:
                color = 'r'
            elif counter == 2:
                color = 'g'
            elif counter == 3:
                color = 'y'
            elif counter == 4:
                color = 'black'
            else:
                color = 'purple'
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
            #print("Delta t:" + str(delta) + "ms" + " und ist " + str(distance) + " entfernt.")

    def getValues(self):
        for loc in self.locationValues:
            print("Delta t:" + str(loc[0]) + "ms" + " und ist " + str(loc[1]) + " entfernt.")





if __name__ == "__main__":
    test = AudioLocate(4,44100)
    test.mixShift()
    #test.mixShiftSpec(-4410,4) # Set a distance of 34.3m

    #test.mixFail()
    test.autocorr()
    test.show()
    test.calculate()
    test.getValues()
