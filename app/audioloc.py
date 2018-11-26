from packages import AudioLocate

if __name__ == "__main__":
    test = AudioLocate(4, 44100)

    test.mix_shift()
    # test.mix_shift_spec(-4410, 4)  # Set a distance of 34.3m

    # test.mix_with_failure()
    test.auto_cross_correlate()
    test.show()
    test.calculate()
    # test.play_mix()
    # test.play_sources()
    test.print_locations()

