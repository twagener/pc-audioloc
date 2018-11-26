from AudioTools import *

if __name__ == "__main__":
    test = AudioLocate(4, 44100)
    # test.mixShift()
    test.mixShiftSpec(-4410, 4)  # Set a distance of 34.3m

    # test.mixFail()
    test.autocorr()
    test.show()
    test.calculate()
    test.getValues()