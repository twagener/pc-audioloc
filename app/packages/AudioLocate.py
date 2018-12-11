import time,sys

from scipy import signal
import numpy as np
#import sympy as sp
import sounddevice as sd
import queue,os

class AudioLocate:
    __input_channels: int
    __channels: int
    __samples: int
    __duration: int

    def __init__(self, channels: int = 4, samplerate: int = 44100, duration: int = 1) -> object:
        self.__output_stream = []
        self.__input_stream = []
        self.__blocksize = 2048
        self.__latency = 1
        self.__channels = channels  # amount of speakers
        self.__duration = duration # duration of noise playback
        self.__samplerate = samplerate # samples - simulate listening for 1 seconds at 44100KHz sample rate
        self.__samples = samplerate * self.__duration
        self.__sources = self.generate_source(self.__channels) # generate n noise channels
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
            mix = np.vstack((mix, source)) # each array shape represents one channel => 4 channels = 4,1 array-shape
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
    def fake_pos(self,x:int = 0,y: int = 0):
        try:
            xy = [x,y]
            self.__distances = np.sqrt(np.sum((self.__speaker_locations-xy)**2,axis=1))
            self.fake_recording_pos()
            return True
        except:
            print("No speaker locations found! Please run set_speaker_locations() before."
                   "\n1\t2\n\n5\t6\n\n3\t4")
            return False

    def fake_recording_pos(self) -> bool:
        try:
            delays = [int(x/343*self.__samplerate) for x in self.__distances]
            mix = np.zeros(self.__samples)
            i = 0
            for source in self.__sources:
                mix += 1 / self.__channels * np.roll(source, delays[i])
                i += 1
            self.__recorded = np.vstack((mix))
            return True
        except:
            print("Failed")
            return False

    ### FAKE OUTPUT

    @staticmethod
    def fake_output_times(x: int = 0, y: int = 0) -> tuple:
        t0 = 0
        t1 = 1
        t2 = 1
        t3 = 1

        return (t0,t1,t2,t3)


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

    def __auto_cross_correlate(self) -> None:
        corr = []
        for source in self.__sources:
            # Pegelcheck um die "besten" Schallquellen zu finden

            # crossCorrelation von original mit umgedrehten recorded
            for i in range(self.__input_channels):
                try:
                    corr.append(signal.fftconvolve(source, self.__input_stream[-1][::-1].T[i], mode='same'))
                except ValueError as err:
                    print(err)
        self.__correlation = corr

    def show(self) -> None:
        import matplotlib.pyplot as plt
        try:
            figure, (ax_mixed, *source_plots) = plt.subplots(self.__channels * self.__input_channels + 1, 1)
            ax_mixed.set_title('Recorded noise')
            ax_mixed.plot(self.__input_stream[-1])
            counter = 0
        except:
            print("No recording found!")

        try:
            for i in range(self.__channels * self.__input_channels):
                counter += 1
                source_plots[i].set_title('Cross correlation for source channel ' + str(counter))
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
        except:
            print("No correlation data found to run show()! Please run auto_cross_correlate() before.")

    def __show(self) -> None:
        import matplotlib.pyplot as plt
        try:
            figure, (ax_mixed, *source_plots) = plt.subplots(self.__channels * self.__input_channels + 1, 1)
            ax_mixed.set_title('Recorded noise')
            ax_mixed.plot(self.__recorded)
            counter = 0
        except:
            print("No recording found!")

        try:
            for i in range(self.__channels * self.__input_channels):
                counter += 1
                source_plots[i].set_title('Cross correlation for source channel ' + str(counter))
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
        except:
            print("No correlation data found to run show()! Please run auto_cross_correlate() before.")

    def calculate(self,show: bool = False) -> None:
        self.__location_values = []
        delay_index = []
        delay = []
        try:
            for i in range(len(self.__correlation)):
                value = (self.__samplerate/2-np.argmax(self.__correlation[i]))/self.__samplerate
                delay_index.append((i,value))
                delay.append(value)
            if show:
                for i in delay:
                    print(str((i) * 343 / self.__duration) + "m")
                    self.__distances = [x * 343 / self.__duration for x in delay]
            else:
                self.__distances = [x*343/self.__duration for x in delay]
        except TypeError:
            print("No values to calculate()! Please run auto_cross_correlate() before.")


    def __calculate(self,show: bool = True) -> None:
        self.__auto_cross_correlate()
        self.__location_values = []
        delay_index = []
        delay = []
        try:
            for i in range(len(self.__correlation)):
                value = (self.__samplerate/2-np.argmax(self.__correlation[i]))/self.__samplerate
                delay_index.append((i,value))
                delay.append(value)
            if show:
                for i in delay:
                    print(str((i) * 343 / self.__duration) + "m")
                    self.__distances = [x * 343 / self.__duration for x in delay]
            else:
                self.__distances = [x*343/self.__duration for x in delay]
        except TypeError:
            print("No values to calculate()! Please run auto_cross_correlate() before.")
        self.show()

    ############# Playback and recording ###############

    def get_devices(self):
        return sd.query_devices()

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

    def start_audio_analyses(self):
        self.__play_thread()
        self.__rec_thread()

    def __rec_thread(self):
        try:
            q = queue.Queue()

            def callback(indata, frames, time, status):
                """This is called (from a separate thread) for each audio block."""
                if status:
                    print(status)
                q.put(indata.copy())

            with sd.InputStream(samplerate=self.__samplerate,blocksize=self.__samplerate, device=self.__input_device,
                                channels=self.__input_channels, callback=callback):
                print('#' * 80)
                print('press Ctrl+C to stop the calculating')
                print('#' * 80)
                while True:
                    print("\nCalculating...")
                    self.__input_stream.append(q.get())
                    self.__calculate()

        except KeyboardInterrupt:
            sd.stop()
            print('\nRecording finished: ')

    def __play_thread(self):
        try:
            """
            Playing the mixed noise in a loop
            """
            try:
                print("Playing loop..")
                sd.play(self.__mixed.T, self.__samplerate, device=self.__output_device, loop=True)
            except (sd.PortAudioError) as err:
                print("Something went wrong!", err, "- Check your output settings and set_output_device()")
                print("\nAvailable devices:", self.get_devices())
                print(self.__mixed.shape)
        except KeyboardInterrupt:
            print('\nPlaying finished... ')

    ##### DEBUG STUFF

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
            sd.play(self.__mixed.T, self.__samplerate, device=self.__output_device, blocking=True)  # NEEDED TO TRANSFORM
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

    ##### SETTER

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

    def set_speaker_locations(self,*positions: tuple) -> None:
        if len(positions) != self.__channels:
            raise ValueError("Got "+str(len(positions))+" positions but there are "+str(self.__channels)+" speakers.")
        else:
            positionList = []
            for pos in positions:
                if type(pos) != tuple:
                    raise ValueError("Need tuples for positions - eg. (x1,y1),(x2,y2). "
                                     "First value should be (0,0)!")
                else:
                    positionList.append(pos)
            self.__speaker_locations = np.array(positionList)

    ##### GETTER

    def get_distances(self) -> list:
        try:
            return self.__distances
        except:
            AttributeError("No distances to return. Run calculate().")

    def get_input_stream(self) -> list:
        try:
            return self.__input_stream
        except:
            AttributeError("No inputStream to return. Run calculate().")



if __name__ == "__main__":
    test = AudioLocate(4, samples=44100)
