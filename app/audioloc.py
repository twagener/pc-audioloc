try:
    from .packages import AudioLocate
    #from .packages import KalmanFilter
except:
    from packages import AudioLocate
    #from packages import KalmanFilter

if __name__ == "__main__":


    locator = AudioLocate(channels=4, samplerate=44100)
    locator.set_speaker_locations((-4, -2), (-4, 2), (4, 2), (4, -2))



    #points = MerweScaledSigmaPoints(n=5, alpha=.1, beta=2., kappa=1.)
    # points = JulierSigmaPoints(n=5, kappa=1)

    #speaker_positions = np.array([[-4, -2], [-4, 2], [4, 2], [4, -2]])


    locator.set_output_device(2)
    locator.set_input_device(0)
    locator.mix_hdmi()

    locator.init_kalman()
    locator.start_audio_analyses()
    #locator.play_mix()


    #print(len(locator.get_input_stream()))
    #locator.play_recorded()
    #locator.calculate(show=False,t0=.1)

    #locator.calculate(plotting=True, initial_position_xy=(-0, -1), t_offset=0.1)
    #locator.show()



