try:
    from .packages import AudioLocate
except:
    from packages import AudioLocate

if __name__ == "__main__":
    locator = AudioLocate(channels=4, samplerate=44100, duration=1)
    locator.set_speaker_locations((0, 0), (0, 8), (4, 8), (4, 0))
  #  locator.get_devices()
    locator.set_output_device(2)
    locator.set_input_device(3)
    #locator.mix_hdmi()

    #locator.rec()
    #locator.play_mix()
    #locator.start_audio_analyses()

    print(len(locator.get_input_stream()))
    #locator.play_recorded()

    #locator.fake_pos(3,2)
    #locator.auto_cross_correlate()
    #locator.calculate()
    #print(locator.get_distances())

    locator.fake_pos(7, 3.8)
    locator.auto_cross_correlate()
    locator.calculate()
    #print(locator.get_distances())



    #locator.play_sources()

    #locator.auto_cross_correlate()
    #locator.calculate()
    #locator.show()
    #locator.print_locations()

