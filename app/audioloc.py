from packages import AudioLocate

if __name__ == "__main__":
    locator = AudioLocate(channels=4, samplerate=44100, duration=10)
    locator.set_output_device(2)
    locator.set_input_device(3)
    locator.mix_hdmi()

    locator.playrec()

    #locator.rec()
    #locator.play_mix()
    #locator.play_recorded()

    #locator.fake_input_shift()
    #locator.fake_input_shift_spec(-4410, 4)  # Set a distance of 34.3m

    locator.auto_cross_correlate()
    locator.calculate()



    #test.mix_with_failure()



    #test.play_sources()
    locator.show()
    locator.print_locations()

