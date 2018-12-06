from packages import AudioLocate

if __name__ == "__main__":
    locator = AudioLocate(channels=4, samplerate=44100, duration=1)

    #locator.set_output_device(2)
    #locator.set_input_device(3)
    #locator.mix_hdmi()

    #locator.playrec()

    #locator.rec()
    #locator.play_mix()
    #locator.play_recorded()

    locator.fake_recording_pos()
    #locator.fake_input_shift_spec(+4410, 4)  # Set a distance of 34.3m
    #locator.fake_input()
    locator.fake_pos()


    #locator.play_sources()

    #locator.auto_cross_correlate()
    #locator.calculate()
    #locator.show()
    #locator.print_locations()

