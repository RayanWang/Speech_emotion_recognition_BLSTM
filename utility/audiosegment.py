"""
This module simply exposes a wrapper of a pydub.AudioSegment object.
"""
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import math
import numpy as np
import pickle
import pydub
import os
import random
import scipy.signal as signal
import subprocess
import sys
import tempfile
import warnings
import webrtcvad

MS_PER_S = 1000
S_PER_MIN = 60
MS_PER_MIN = MS_PER_S * S_PER_MIN

def deprecated(func):
    """
    Deprecator decorator.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__), category=DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    return new_func

class AudioSegment:
    """
    This class is a wrapper for a pydub.AudioSegment that provides additional methods.
    """

    def __init__(self, pydubseg, name):
        self.seg = pydubseg
        self.name = name

    def __getattr__(self, attr):
        orig_attr = self.seg.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                if result == self.seg:
                    return self
                elif type(result) == pydub.AudioSegment:
                    return AudioSegment(result, self.name)
                else:
                    return  result
            return hooked
        else:
            return orig_attr

    def __len__(self):
        return len(self.seg)

    def __eq__(self, other):
        return self.seg == other

    def __ne__(self, other):
        return self.seg != other

    def __iter__(self, other):
        return (x for x in self.seg)

    def __getitem__(self, millisecond):
        return AudioSegment(self.seg[millisecond], self.name)

    def __add__(self, arg):
        if type(arg) == AudioSegment:
            self.seg._data = self.seg._data + arg.seg._data
        else:
            self.seg = self.seg + arg
        return self

    def __radd__(self, rarg):
        return self.seg.__radd__(rarg)

    def __repr__(self):
        return str(self)

    def __str__(self):
        s = "%s: %s channels, %s bit, sampled @ %s kHz, %.3fs long" %\
            (self.name, str(self.channels), str(self.sample_width * 8),\
             str(self.frame_rate / 1000.0), self.duration_seconds)
        return s

    def __sub__(self, arg):
        if type(arg) == AudioSegment:
            self.seg = self.seg - arg.seg
        else:
            self.seg = self.seg - arg
        return self

    def __mul__(self, arg):
        if type(arg) == AudioSegment:
            self.seg = self.seg * arg.seg
        else:
            self.seg = self.seg * arg
        return self

    @property
    def spl(self):
        """
        Sound Pressure Level - defined as 20 * log10(abs(value)).

        Returns a numpy array of SPL dB values.
        """
        return 20.0 * np.log10(np.abs(self.to_numpy_array() + 1E-9))

    def _bandpass_filter(self, data, low, high, fs, order=5):
        """
        :param data: The data (numpy array) to be filtered.
        :param low: The low cutoff in Hz.
        :param high: The high cutoff in Hz.
        :param fs: The sample rate (in Hz) of the data.
        :param order: The order of the filter. The higher the order, the tighter the roll-off.
        :returns: Filtered data (numpy array).
        """
        nyq = 0.5 * fs
        low = low / nyq
        high = high / nyq
        b, a = signal.butter(order, [low, high], btype='band')
        y = signal.lfilter(b, a, data)
        return y

    def auditory_scene_analysis(self):
        """
        Algorithm based on paper: Auditory Segmentation Based on Onset and Offset Analysis,
        by Hu and Wang, 2007.
        """
        def lowpass_filter(data, cutoff, fs, order=5):
            """
            """
            nyq = 0.5 * fs
            normal_cutoff = cutoff / nyq
            b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
            y = signal.lfilter(b, a, data)
            return y

        import matplotlib.pyplot as plt

        def visualize_time_domain(seg, title=""):
            plt.plot(seg)
            plt.title(title)
            plt.show()
            plt.clf()

        def visualize(spect, frequencies, title=""):
            i = 0
            for freq, (index, row) in zip(frequencies[::-1], enumerate(spect[::-1, :])):
                plt.subplot(spect.shape[0], 1, index + 1)
                if i == 0:
                    plt.title(title)
                    i += 1
                plt.ylabel("{0:.0f}".format(freq))
                plt.plot(row)
            plt.show()
            plt.clf()

        # Normalize self into 25dB average SPL
        normalized = self.normalize_spl_by_average(db=25)
        visualize_time_domain(normalized.to_numpy_array(), "Normalized")
        # Do a band-pass filter in each frequency
        data = normalized.to_numpy_array()
        start_frequency = 50
        stop_frequency = 8000
        start = np.log10(start_frequency)
        stop = np.log10(stop_frequency)
        frequencies = np.logspace(start, stop, num=10, endpoint=True, base=10.0)
        print("Dealing with the following frequencies:", frequencies)
        rows = [self._bandpass_filter(data, freq*0.8, freq*1.2, self.frame_rate) for freq in frequencies]
        rows = np.array(rows)
        spect = np.vstack(rows)
        visualize(spect, frequencies, "After bandpass filtering (cochlear model)")

        # Half-wave rectify each frequency channel
        spect[spect < 0] = 0
        visualize(spect, frequencies, "After half-wave rectification in each frequency")

        # Low-pass filter each frequency channel
        spect = np.apply_along_axis(lowpass_filter, 1, spect, 30, self.frame_rate, 6)
        visualize(spect, frequencies, "After low-pass filtering in each frequency")

        # Downsample each frequency to 400 Hz
        downsample_freq_hz = 400
        if self.frame_rate > downsample_freq_hz:
            step = int(round(self.frame_rate / downsample_freq_hz))
            spect = spect[:, ::step]
        visualize(spect, frequencies, "After downsampling in each frequency")

        # Now you have the temporal envelope of each frequency channel

        # Smoothing
        scales = [(6, 1/4), (6, 1/14), (1/2, 1/14)]
        thetas = [0.95,     0.95,      0.85]
        ## For each (sc, st) scale, smooth across time using st, then across frequency using sc
        gaussian = lambda x, mu, sig: np.exp(-np.power(x - mu, 2.0) / (2 * np.power(sig, 2.0)))
        gaussian_kernel = lambda sig: gaussian(np.linspace(-10, 10, len(frequencies) / 2), 0, sig)
        spectrograms = []
        for sc, st in scales:
            time_smoothed = np.apply_along_axis(lowpass_filter, 1, spect, 1/st, downsample_freq_hz, 6)
            visualize(time_smoothed, frequencies, "After time smoothing with scale: " + str(st))
            freq_smoothed = np.apply_along_axis(np.convolve, 0, spect, gaussian_kernel(sc))
            spectrograms.append(freq_smoothed)
            visualize(freq_smoothed, frequencies, "After time and frequency smoothing with scales (freq) " + str(sc) + " and (time) " + str(st))
        ## Now we have a set of scale-space spectrograms of different scales (sc, st)

        # Onset/Offset Detection and Matching
        def theta_on(spect):
            return np.nanmean(spect) + np.nanstd(spect)

        def compute_peaks_or_valleys_of_first_derivative(s, do_peaks=True):
            """
            Takes a spectrogram and returns a 2D array of the form:

            0 0 0 1 0 0 1 0 0 0 1   <-- Frequency 0
            0 0 1 0 0 0 0 0 0 1 0   <-- Frequency 1
            0 0 0 0 0 0 1 0 1 0 0   <-- Frequency 2
            *** Time axis *******

            Where a 1 means that the value in that time bin in the spectrogram corresponds to
            a peak/valley in the first derivative.
            """
            gradient = np.nan_to_num(np.apply_along_axis(np.gradient, 1, s), copy=False)
            half_window = 4
            if do_peaks:
                indexes = [signal.argrelextrema(gradient[i, :], np.greater, order=half_window) for i in range(gradient.shape[0])]
            else:
                indexes = [signal.argrelextrema(gradient[i, :], np.less, order=half_window) for i in range(gradient.shape[0])]
            extrema = np.zeros(s.shape)
            for row_index, index_array in enumerate(indexes):
                # Each index_array is a list of indexes corresponding to all the extrema in a given row
                for col_index in index_array:
                    extrema[row_index, col_index] = 1
            return extrema

        for spect, (sc, st) in zip(spectrograms, scales):
            # Compute sudden upward changes in spect, these are onsets of events
            onsets = compute_peaks_or_valleys_of_first_derivative(spect)
            # Compute sudden downward changes in spect, these are offsets of events
            offsets = compute_peaks_or_valleys_of_first_derivative(spect, do_peaks=False)
            print("TOTAL ONSETS:", np.sum(onsets, axis=1))
            print("TOTAL OFFSETS:", np.sum(offsets, axis=1))
            exit()

            # onsets and offsets are 2D arrays

            ## Determine the offset time for each onset:
            ### If t_on[c, i] represents the time of the ith onset in frequency channel c, the corresponding offset
            ###     must occur between t_on[c, i] and t_on[c, i+1]
            ### If there are more than one offsets candidates in this range, choose the one with largest intensity decrease.
            ## Create onset/offset fronts by connecting onsets across frequency channels (connect two onsets
            ##      if they occur within 20ms of each other). Start over whenever a frequency band does not contain an offset
            ##      in this range. Do the same procedure for offsets. Now you have onset and offset fronts.
            ## Now hook up the onsets with the offsets to form segments:
            ##      For each onset front, (t_on[c, i1, t_on[c + 1, i2], ..., t_on[c + m - 1, im]):
            ##          matching_offsets = (t_off[c, i1], t_off[c + 1, i2], ..., t_off[c + m - 1, im])
            ##          Get all offset fronts which contain at least one of offset time found in matching_offsets
            ##          Among these offset fronts, the one that crosses the most of matching_offsets is chosen,
            ##          - call this offset front: matching_offset_front
            ##          Update all t_offs in matching_offsets whose 'c's are in matching_offset_front to be 'matched', and
            ##          - update their times to the corresponding channel offset in matching_offset_front.
            ##          If all t_offs in matching_offsets are 'matched', continue to next onset front
            ## Now go through all the segments you have created and break them up along frequencies if the temporal
            ##      envelopes don't match well enough. That is, if we have two adjacent channels c and c+1, and they
            ##      are part of the same segment as determined above, break this segment into two along these lines
            ##      if the correlation between them is below theta_c. Theta_c is thetas[i] where i depends on the scale.

        # Multiscale Integration
        ##
        ## TODO

    def detect_voice(self, prob_detect_voice=0.5):
        """
        Returns self as a list of tuples:
        [('v', voiced segment), ('u', unvoiced segment), (etc.)]

        The overall order of the AudioSegment is preserved.

        :param prob_detect_voice: The raw probability that any random 20ms window of the audio file
                                  contains voice.
        :returns: The described list.
        """
        assert self.frame_rate in (48000, 32000, 16000, 8000), "Try resampling to one of the allowed frame rates."
        assert self.sample_width == 2, "Try resampling to 16 bit."
        assert self.channels == 1, "Try resampling to one channel."

        class model_class:
            def __init__(self, aggressiveness):
                self.v = webrtcvad.Vad(int(aggressiveness))

            def predict(self, vector):
                if self.v.is_speech(vector.raw_data, vector.frame_rate):
                    return 1
                else:
                    return 0

        model = model_class(aggressiveness=1)
        pyesno = 0.3  # Probability of the next 20 ms being unvoiced given that this 20 ms was voiced
        pnoyes = 0.2  # Probability of the next 20 ms being voiced given that this 20 ms was unvoiced
        p_realyes_outputyes = 0.4  # WebRTCVAD has a very high FP rate - just because it says yes, doesn't mean much
        p_realyes_outputno  = 0.05  # If it says no, we can be very certain that it really is a no
        p_yes_raw = prob_detect_voice
        filtered = self.detect_event(model=model,
                                     ms_per_input=20,
                                     transition_matrix=(pyesno, pnoyes),
                                     model_stats=(p_realyes_outputyes, p_realyes_outputno),
                                     event_length_s=0.25,
                                     prob_raw_yes=p_yes_raw)
        ret = []
        for tup in filtered:
            t = ('v', tup[1]) if tup[0] == 'y' else ('u', tup[1])
            ret.append(t)
        return ret

    def dice(self, seconds, zero_pad=False):
        """
        Cuts the AudioSegment into `seconds` segments (at most). So for example, if seconds=10,
        this will return a list of AudioSegments, in order, where each one is at most 10 seconds
        long. If `zero_pad` is True, the last item AudioSegment object will be zero padded to result
        in `seconds` seconds.

        :param seconds: The length of each segment in seconds. Can be either a float/int, in which case
                        `self.duration_seconds` / `seconds` are made, each of `seconds` length, or a
                        list-like can be given, in which case the given list must sum to
                        `self.duration_seconds` and each segment is specified by the list - e.g.
                        the 9th AudioSegment in the returned list will be `seconds[8]` seconds long.
        :param zero_pad: Whether to zero_pad the final segment if necessary. Ignored if `seconds` is
                         a list-like.
        :returns: A list of AudioSegments, each of which is the appropriate number of seconds long.
        :raises: ValueError if a list-like is given for `seconds` and the list's durations do not sum
                 to `self.duration_seconds`.
        """
        try:
            total_s = sum(seconds)
            if not (self.duration_seconds <= total_s + 1 and self.duration_seconds >= total_s - 1):
                raise ValueError("`seconds` does not sum to within one second of the duration of this AudioSegment.\
                                 given total seconds: %s and self.duration_seconds: %s" % (total_s, self.duration_seconds))
            starts = []
            stops = []
            time_ms = 0
            for dur in seconds:
                starts.append(time_ms)
                time_ms += dur * MS_PER_S
                stops.append(time_ms)
            zero_pad = False
        except TypeError:
            # `seconds` is not a list
            starts = range(0, int(round(self.duration_seconds * MS_PER_S)), int(round(seconds * MS_PER_S)))
            stops = (min(self.duration_seconds * MS_PER_S, start + seconds * MS_PER_S) for start in starts)
        outs = [self[start:stop] for start, stop in zip(starts, stops)]
        out_lens = [out.duration_seconds for out in outs]
        # Check if our last slice is within one ms of expected - if so, we don't need to zero pad
        if zero_pad and not (out_lens[-1] <= seconds * MS_PER_S + 1 and out_lens[-1] >= seconds * MS_PER_S - 1):
            num_zeros = self.frame_rate * (seconds * MS_PER_S - out_lens[-1])
            outs[-1] = outs[-1].zero_extend(num_samples=num_zeros)
        return outs

    def detect_event(self, model, ms_per_input, transition_matrix, model_stats, event_length_s,
                     start_as_yes=False, prob_raw_yes=0.5):
        """
        A list of tuples of the form [('n', AudioSegment), ('y', AudioSegment), etc.] is returned, where tuples
        of the form ('n', AudioSegment) are the segments of sound where the event was not detected,
        while ('y', AudioSegment) tuples were the segments of sound where the event was detected.

        .. code-block:: python

            # Example usage
            import audiosegment
            import keras
            import keras.models
            import numpy as np
            import sys

            class Model:
                def __init__(self, modelpath):
                    self.model = keras.models.load_model(modelpath)

                def predict(self, seg):
                    _bins, fft_vals = seg.fft()
                    fft_vals = np.abs(fft_vals) / len(fft_vals)
                    predicted_np_form = self.model.predict(np.array([fft_vals]), batch_size=1)
                    prediction_as_int = int(round(predicted_np_form[0][0]))
                    return prediction_as_int

            modelpath = sys.argv[1]
            wavpath = sys.argv[2]
            model = Model(modelpath)
            seg = audiosegment.from_file(wavpath).resample(sample_rate_Hz=32000, sample_width=2, channels=1)
            pyes_to_no = 0.3  # The probability of one 30 ms sample being an event, and the next one not
            pno_to_yes = 0.2  # The probability of one 30 ms sample not being an event, and the next one yes
            ptrue_pos_rate = 0.8  # The true positive rate (probability of a predicted yes being right)
            pfalse_neg_rate = 0.3  # The false negative rate (probability of a predicted no being wrong)
            raw_prob = 0.7  # The raw probability of seeing the event in any random 30 ms slice of this file
            events = seg.detect_event(model, ms_per_input=30, transition_matrix=[pyes_to_no, pno_to_yes],
                                      model_stats=[ptrue_pos_rate, pfalse_neg_rate], event_length_s=0.25,
                                      prob_raw_yes=raw_prob)
            nos = [event[1] for event in events if event[0] == 'n']
            yeses = [event[1] for event in events if event[0] == 'y']
            if len(nos) > 1:
                notdetected = nos[0].reduce(nos[1:])
                notdetected.export("notdetected.wav", format="WAV")
            if len(yeses) > 1:
                detected = yeses[0].reduce(yeses[1:])
                detected.export("detected.wav", format="WAV")


        :param model:               The model. The model must have a predict() function which takes an AudioSegment
                                    of `ms_per_input` number of ms and which outputs 1 if the audio event is detected
                                    in that input, and 0 if not. Make sure to resample the AudioSegment to the right
                                    values before calling this function on it.

        :param ms_per_input:        The number of ms of AudioSegment to be fed into the model at a time. If this does not
                                    come out even, the last AudioSegment will be zero-padded.

        :param transition_matrix:   An iterable of the form: [p(yes->no), p(no->yes)]. That is, the probability of moving
                                    from a 'yes' state to a 'no' state and the probability of vice versa.

        :param model_stats:         An iterable of the form: [p(reality=1|output=1), p(reality=1|output=0)]. That is,
                                    the probability of the ground truth really being a 1, given that the model output a 1,
                                    and the probability of the ground truth being a 1, given that the model output a 0.

        :param event_length_s:      The typical duration of the event you are looking for in seconds (can be a float).

        :param start_as_yes:        If True, the first `ms_per_input` will be in the 'y' category. Otherwise it will be
                                    in the 'n' category.

        :param prob_raw_yes:        The raw probability of finding the event in any given `ms_per_input` vector.

        :returns:                   A list of tuples of the form [('n', AudioSegment), ('y', AudioSegment), etc.],
                                    where over the course of the list, the AudioSegment in tuple 3 picks up
                                    where the one in tuple 2 left off.

        :raises:                    ValueError if `ms_per_input` is negative or larger than the number of ms in this
                                    AudioSegment; if `transition_matrix` or `model_stats` do not have a __len__ attribute
                                    or are not length 2; if the values in `transition_matrix` or `model_stats` are not
                                    in the closed interval [0.0, 1.0].
        """
        if ms_per_input < 0 or ms_per_input / MS_PER_S > self.duration_seconds:
            raise ValueError("ms_per_input cannot be negative and cannot be longer than the duration of the AudioSegment."\
                             " The given value was " + str(ms_per_input))
        elif not hasattr(transition_matrix, "__len__") or len(transition_matrix) != 2:
            raise ValueError("transition_matrix must be an iterable of length 2.")
        elif not hasattr(model_stats, "__len__") or len(model_stats) != 2:
            raise ValueError("model_stats must be an iterable of length 2.")
        elif any([True for prob in transition_matrix if prob > 1.0 or prob < 0.0]):
            raise ValueError("Values in transition_matrix are probabilities, and so must be in the range [0.0, 1.0].")
        elif any([True for prob in model_stats if prob > 1.0 or prob < 0.0]):
            raise ValueError("Values in model_stats are probabilities, and so must be in the range [0.0, 1.0].")
        elif prob_raw_yes > 1.0 or prob_raw_yes < 0.0:
            raise ValueError("`prob_raw_yes` is a probability, and so must be in the range [0.0, 1.0]")

        # Get the yeses or nos for when the filter is triggered (when the event is on/off)
        filter_indices = [yes_or_no for yes_or_no in self._get_filter_indices(start_as_yes,
                                                                              prob_raw_yes,
                                                                              ms_per_input,
                                                                              model,
                                                                              transition_matrix,
                                                                              model_stats)]
        # Run a homogeneity filter over the values to make local regions more self-similar (reduce noise)
        ret = self._homogeneity_filter(filter_indices, window_size=int(round(0.25 * MS_PER_S / ms_per_input)))
        # Group the consecutive ones together
        ret = self._group_filter_values(ret, ms_per_input)
        # Take the groups and turn them into AudioSegment objects
        real_ret = self._reduce_filtered_segments(ret)

        return real_ret

    def _get_filter_indices(self, start_as_yes, prob_raw_yes, ms_per_input, model, transition_matrix, model_stats):
        """
        This has been broken out of the `filter` function to reduce cognitive load.
        """
        filter_triggered = 1 if start_as_yes else 0
        prob_raw_no = 1.0 - prob_raw_yes
        for segment, _timestamp in self.generate_frames_as_segments(ms_per_input):
            yield filter_triggered
            observation = int(round(model.predict(segment)))
            assert observation == 1 or observation == 0, "The given model did not output a 1 or a 0, output: "\
                   + str(observation)
            prob_hyp_yes_given_last_hyp = 1.0 - transition_matrix[0] if filter_triggered else transition_matrix[1]
            prob_hyp_no_given_last_hyp  = transition_matrix[0] if filter_triggered else 1.0 - transition_matrix[1]
            prob_hyp_yes_given_data = model_stats[0] if observation == 1 else model_stats[1]
            prob_hyp_no_given_data = 1.0 - model_stats[0] if observation == 1 else 1.0 - model_stats[1]
            hypothesis_yes = prob_raw_yes * prob_hyp_yes_given_last_hyp * prob_hyp_yes_given_data
            hypothesis_no  = prob_raw_no * prob_hyp_no_given_last_hyp  * prob_hyp_no_given_data
            # make a list of ints - each is 0 or 1. The number of 1s is hypotheis_yes * 100
            # the number of 0s is hypothesis_no * 100
            distribution = [1 for i in range(int(round(hypothesis_yes * 100)))]
            distribution.extend([0 for i in range(int(round(hypothesis_no * 100)))])
            # shuffle
            random.shuffle(distribution)
            filter_triggered = random.choice(distribution)

    def _group_filter_values(self, filter_indices, ms_per_input):
        """
        This has been broken out of the `filter` function to reduce cognitive load.
        """
        ret = []
        for filter_value, (_segment, timestamp) in zip(filter_indices, self.generate_frames_as_segments(ms_per_input)):
            if filter_value == 1:
                if len(ret) > 0 and ret[-1][0] == 'n':
                    ret.append(['y', timestamp])  # The last one was different, so we create a new one
                elif len(ret) > 0 and ret[-1][0] == 'y':
                    ret[-1][1] = timestamp  # The last one was the same as this one, so just update the timestamp
                else:
                    ret.append(['y', timestamp])  # This is the first one
            else:
                if len(ret) > 0 and ret[-1][0] == 'n':
                    ret[-1][1] = timestamp
                elif len(ret) > 0 and ret[-1][0] == 'y':
                    ret.append(['n', timestamp])
                else:
                    ret.append(['n', timestamp])
        return ret

    def _homogeneity_filter(self, ls, window_size):
        """
        This has been broken out of the `filter` function to reduce cognitive load.

        ls is a list of 1s or 0s for when the filter is on or off
        """
        k = window_size
        i = k
        while i <= len(ls) - k:
            # Get a window of k items
            window = [ls[i + j] for j in range(k)]
            # Change the items in the window to be more like the mode of that window
            mode = 1 if sum(window) >= k / 2 else 0
            for j in range(k):
                ls[i+j] = mode
            i += k
        return ls

    def _reduce_filtered_segments(self, ret):
        """
        This has been broken out of the `filter` function to reduce cognitive load.
        """
        real_ret = []
        for i, (this_yesno, next_timestamp) in enumerate(ret):
            if i > 0:
                _next_yesno, timestamp = ret[i - 1]
            else:
                timestamp = 0

            data = self[timestamp * MS_PER_S:next_timestamp * MS_PER_S].raw_data
            seg = AudioSegment(pydub.AudioSegment(data=data, sample_width=self.sample_width,
                                                  frame_rate=self.frame_rate, channels=self.channels), self.name)
            real_ret.append((this_yesno, seg))
        return real_ret

    def filter_silence(self, duration_s=1, threshold_percentage=1, console_output=False):
        """
        Returns a copy of this AudioSegment, but whose silence has been removed.

        .. note:: This method requires that you have the program 'sox' installed.

        .. warning:: This method uses the program 'sox' to perform the task. While this is very fast for a single
                     function call, the IO may add up for a large numbers of AudioSegment objects.

        :param duration_s: The number of seconds of "silence" that must be present in a row to
                           be stripped.
        :param threshold_percentage: Silence is defined as any samples whose absolute value is below
                                     `threshold_percentage * max(abs(samples in this segment))`.
        :param console_output: If True, will pipe all sox output to the console.
        :returns: A copy of this AudioSegment, but whose silence has been removed.
        """
        tmp = tempfile.NamedTemporaryFile()
        othertmp = tempfile.NamedTemporaryFile()
        self.export(tmp.name, format="WAV")
        command = "sox " + tmp.name + " -t wav " + othertmp.name + " silence -l 1 0.1 "\
                   + str(threshold_percentage) + "% -1 " + str(float(duration_s)) + " " + str(threshold_percentage) + "%"
        stdout = stderr = subprocess.PIPE if console_output else subprocess.DEVNULL
        res = subprocess.run(command.split(' '), stdout=stdout, stderr=stderr)
        assert res.returncode == 0, "Sox did not work as intended, or perhaps you don't have Sox installed?"
        other = AudioSegment(pydub.AudioSegment.from_wav(othertmp.name), self.name)
        tmp.close()
        othertmp.close()
        return other

    def fft(self, start_s=None, duration_s=None, start_sample=None, num_samples=None, zero_pad=False):
        """
        Transforms the indicated slice of the AudioSegment into the frequency domain and returns the bins
        and the values.

        If neither `start_s` or `start_sample` is specified, the first sample of the slice will be the first sample
        of the AudioSegment.

        If neither `duration_s` or `num_samples` is specified, the slice will be from the specified start
        to the end of the segment.

        .. code-block:: python

            # Example for plotting the FFT using this function
            import matplotlib.pyplot as plt
            import numpy as np

            seg = audiosegment.from_file("furelise.wav")
            # Just take the first 3 seconds
            hist_bins, hist_vals = seg[1:3000].fft()
            hist_vals_real_normed = np.abs(hist_vals) / len(hist_vals)
            plt.plot(hist_bins / 1000, hist_vals_real_normed)
            plt.xlabel("kHz")
            plt.ylabel("dB")
            plt.show()

        .. image:: images/fft.png

        :param start_s: The start time in seconds. If this is specified, you cannot specify `start_sample`.
        :param duration_s: The duration of the slice in seconds. If this is specified, you cannot specify `num_samples`.
        :param start_sample: The zero-based index of the first sample to include in the slice.
                             If this is specified, you cannot specify `start_s`.
        :param num_samples: The number of samples to include in the slice. If this is specified, you cannot
                            specify `duration_s`.
        :param zero_pad: If True and the combination of start and duration result in running off the end of
                         the AudioSegment, the end is zero padded to prevent this.
        :returns: np.ndarray of frequencies, np.ndarray of amount of each frequency
        :raises: ValueError If `start_s` and `start_sample` are both specified and/or if both `duration_s` and
                            `num_samples` are specified.
        """
        if start_s is not None and start_sample is not None:
            raise ValueError("Only one of start_s and start_sample can be specified.")
        if duration_s is not None and num_samples is not None:
            raise ValueError("Only one of duration_s and num_samples can be specified.")
        if start_s is None and start_sample is None:
            start_sample = 0
        if duration_s is None and num_samples is None:
            num_samples = len(self.get_array_of_samples()) - int(start_sample)

        if duration_s is not None:
            num_samples = int(round(duration_s * self.frame_rate))
        if start_s is not None:
            start_sample = int(round(start_s * self.frame_rate))

        end_sample = start_sample + num_samples  # end_sample is excluded
        if end_sample > len(self.get_array_of_samples()) and not zero_pad:
            raise ValueError("The combination of start and duration will run off the end of the AudioSegment object.")
        elif end_sample > len(self.get_array_of_samples()) and zero_pad:
            arr = np.array(self.get_array_of_samples())
            zeros = np.zeros(end_sample - len(arr))
            arr = np.append(arr, zeros)
        else:
            arr = np.array(self.get_array_of_samples())

        audioslice = np.array(arr[start_sample:end_sample])
        fft_result = np.fft.fft(audioslice)[range(int(round(num_samples/2)) + 1)]
        bins = np.arange(0, int(round(num_samples/2)) + 1, 1.0) * (self.frame_rate / num_samples)
        return bins, fft_result

    def generate_frames(self, frame_duration_ms, zero_pad=True):
        """
        Yields self's data in chunks of frame_duration_ms.

        This function adapted from pywebrtc's example [https://github.com/wiseman/py-webrtcvad/blob/master/example.py].

        :param frame_duration_ms: The length of each frame in ms.
        :param zero_pad: Whether or not to zero pad the end of the AudioSegment object to get all
                         the audio data out as frames. If not, there may be a part at the end
                         of the Segment that is cut off (the part will be <= `frame_duration_ms` in length).
        :returns: A Frame object with properties 'bytes (the data)', 'timestamp (start time)', and 'duration'.
        """
        Frame = collections.namedtuple("Frame", "bytes timestamp duration")

        # (samples/sec) * (seconds in a frame) * (bytes/sample)
        bytes_per_frame = int(self.frame_rate * (frame_duration_ms / 1000) * self.sample_width)
        offset = 0  # where we are so far in self's data (in bytes)
        timestamp = 0.0  # where we are so far in self (in seconds)
        # (bytes/frame) * (sample/bytes) * (sec/samples)
        frame_duration_s = (bytes_per_frame / self.frame_rate) / self.sample_width
        while offset + bytes_per_frame < len(self.raw_data):
            yield Frame(self.raw_data[offset:offset + bytes_per_frame], timestamp, frame_duration_s)
            timestamp += frame_duration_s
            offset += bytes_per_frame

        if zero_pad:
            rest = self.raw_data[offset:]
            if len(rest) >= bytes_per_frame and len(rest) % bytes_per_frame:
                zeros = bytes(bytes_per_frame - len(rest))
                yield Frame(rest + zeros, timestamp, frame_duration_s)

    def generate_frames_as_segments(self, frame_duration_ms, zero_pad=True):
        """
        Does the same thing as `generate_frames`, but yields tuples of (AudioSegment, timestamp) instead of Frames.
        """
        for frame in self.generate_frames(frame_duration_ms, zero_pad=zero_pad):
            seg = AudioSegment(pydub.AudioSegment(data=frame.bytes, sample_width=self.sample_width,
                               frame_rate=self.frame_rate, channels=self.channels), self.name)
            yield seg, frame.timestamp

    def normalize_spl_by_average(self, db):
        """
        Normalize the values in the AudioSegment so that its average dB value
        is `db`.

        The dB of a value is calculated as 20 * log10(abs(value + 1E-9)).

        :param db: The decibels to normalize average to.
        :returns: A new AudioSegment object whose values are changed so that their
                  average is `db`.
        """
        def inverse_spl(val):
            """Calculates the (positive) 'PCM' value for the given SPl val"""
            return 10 ** (val / 20.0)

        # Convert dB into 'PCM'
        db_pcm = inverse_spl(db)
        # Calculate current 'PCM' average
        curavg = np.abs(np.mean(self.to_numpy_array()))
        # Calculate ratio of dB_pcm / curavg_pcm
        ratio = db_pcm / curavg
        # Multiply all values by ratio
        dtype_dict = {1: np.int8, 2: np.int16, 4: np.int32}
        dtype = dtype_dict[self.sample_width]
        new_seg = from_numpy_array(np.array(self.to_numpy_array() * ratio, dtype=dtype), self.frame_rate)
        # Check SPL average to see if we are right
        #assert math.isclose(np.mean(new_seg.spl), db), "new = " + str(np.mean(new_seg.spl)) + " != " + str(db)
        return new_seg

    def reduce(self, others):
        """
        Reduces others into this one by concatenating all the others onto this one and
        returning the result. Does not modify self, instead, makes a copy and returns that.

        :param others: The other AudioSegment objects to append to this one.
        :returns: The concatenated result.
        """
        ret = AudioSegment(self.seg, self.name)
        selfdata = [self.seg._data]
        otherdata = [o.seg._data for o in others]
        ret.seg._data = b''.join(selfdata + otherdata)

        return ret

    def resample(self, sample_rate_Hz=None, sample_width=None, channels=None, console_output=False):
        """
        Returns a new AudioSegment whose data is the same as this one, but which has been resampled to the
        specified characteristics. Any parameter left None will be unchanged.

        .. note:: This method requires that you have the program 'sox' installed.

        .. warning:: This method uses the program 'sox' to perform the task. While this is very fast for a single
                     function call, the IO may add up for a large numbers of AudioSegment objects.

        :param sample_rate_Hz: The new sample rate in Hz.
        :param sample_width: The new sample width in bytes, so sample_width=2 would correspond to 16 bit (2 byte) width.
        :param channels: The new number of channels.
        :param console_output: Will print the output of sox to the console if True.
        :returns: The newly sampled AudioSegment.
        """
        if sample_rate_Hz is None:
            sample_rate_Hz = self.frame_rate
        if sample_width is None:
            sample_width = self.sample_width
        if channels is None:
            channels = self.channels

        infile, outfile = tempfile.NamedTemporaryFile(), tempfile.NamedTemporaryFile()
        self.export(infile.name, format="wav")
        command = "sox " + infile.name + " -b" + str(sample_width * 8) + " -r " + str(sample_rate_Hz) + " -t wav " + outfile.name + " channels " + str(channels)
        # stdout = stderr = subprocess.PIPE if console_output else subprocess.DEVNULL
        # res = subprocess.run(command.split(' '), stdout=stdout, stderr=stderr)
        res = subprocess.call(command.split(' '))
        if res:
            raise subprocess.CalledProcessError(res, cmd=command)
        other = AudioSegment(pydub.AudioSegment.from_wav(outfile.name), self.name)
        infile.close()
        outfile.close()
        return other

    def serialize(self):
        """
        Serializes into a bytestring.

        :returns: An object of type Bytes.
        """
        d = {'name': self.name, 'seg': pickle.dumps(self.seg, protocol=-1)}
        return pickle.dumps(d, protocol=-1)

    def spectrogram(self, start_s=None, duration_s=None, start_sample=None, num_samples=None,
                    window_length_s=None, window_length_samples=None, overlap=0.5):
        """
        Does a series of FFTs from `start_s` or `start_sample` for `duration_s` or `num_samples`.
        Effectively, transforms a slice of the AudioSegment into the frequency domain across different
        time bins.

        .. code-block:: python

            # Example for plotting a spectrogram using this function
            import audiosegment
            import matplotlib.pyplot as plt

            #...
            seg = audiosegment.from_file("somebodytalking.wav")
            freqs, times, amplitudes = seg.spectrogram(window_length_s=0.03, overlap=0.5)
            amplitudes = 10 * np.log10(amplitudes + 1e-9)

            # Plot
            plt.pcolormesh(times, freqs, amplitudes)
            plt.xlabel("Time in Seconds")
            plt.ylabel("Frequency in Hz")
            plt.show()

        .. image:: images/spectrogram.png

        :param start_s: The start time. Starts at the beginning if neither this nor `start_sample` is specified.
        :param duration_s: The duration of the spectrogram in seconds. Goes to the end if neither this nor
                           `num_samples` is specified.
        :param start_sample: The index of the first sample to use. Starts at the beginning if neither this nor
                             `start_s` is specified.
        :param num_samples: The number of samples in the spectrogram. Goes to the end if neither this nor
                            `duration_s` is specified.
        :param window_length_s: The length of each FFT in seconds. If the total number of samples in the spectrogram
                                is not a multiple of the window length in samples, the last window will be zero-padded.
        :param window_length_samples: The length of each FFT in number of samples. If the total number of samples in the
                                spectrogram is not a multiple of the window length in samples, the last window will
                                be zero-padded.
        :param overlap: The fraction of each window to overlap.
        :returns: Three np.ndarrays: The frequency values in Hz (the y-axis in a spectrogram), the time values starting
                  at start time and then increasing by `duration_s` each step (the x-axis in a spectrogram), and
                  the dB of each time/frequency bin as a 2D array of shape [len(frequency values), len(duration)].
        :raises ValueError: If `start_s` and `start_sample` are both specified, if `duration_s` and `num_samples` are both
                            specified, if the first window's duration plus start time lead to running off the end
                            of the AudioSegment, or if `window_length_s` and `window_length_samples` are either
                            both specified or if they are both not specified.
        """
        if start_s is not None and start_sample is not None:
            raise ValueError("Only one of start_s and start_sample may be specified.")
        if duration_s is not None and num_samples is not None:
            raise ValueError("Only one of duration_s and num_samples may be specified.")
        if window_length_s is not None and window_length_samples is not None:
            raise ValueError("Only one of window_length_s and window_length_samples may be specified.")
        if window_length_s is None and window_length_samples is None:
            raise ValueError("You must specify a window length, either in window_length_s or in window_length_samples.")

        if start_s is None and start_sample is None:
            start_sample = 0
        if duration_s is None and num_samples is None:
            num_samples = len(self.get_array_of_samples()) - int(start_sample)

        if duration_s is not None:
            num_samples = int(round(duration_s * self.frame_rate))
        if start_s is not None:
            start_sample = int(round(start_s * self.frame_rate))

        if window_length_s is not None:
            window_length_samples = int(round(window_length_s * self.frame_rate))

        if start_sample + num_samples > len(self.get_array_of_samples()):
            raise ValueError("The combination of start and duration will run off the end of the AudioSegment object.")

        f, t, sxx = signal.spectrogram(self.to_numpy_array(), self.frame_rate, scaling='spectrum', nperseg=window_length_samples,
                                             noverlap=int(round(overlap * window_length_samples)),
                                             mode='magnitude')
        return f, t, sxx

    def to_numpy_array(self):
        """
        Convenience function for `np.array(self.get_array_of_samples())` while
        keeping the appropriate dtype.
        """
        dtype_dict = {
                        1: np.int8,
                        2: np.int16,
                        4: np.int32
                     }
        dtype = dtype_dict[self.sample_width]
        return np.array(self.get_array_of_samples(), dtype=dtype)

    @deprecated
    def trim_to_minutes(self, strip_last_seconds=False):
        """
        Returns a list of minute-long (at most) Segment objects.

        .. note:: This function has been deprecated. Use the `dice` function instead.

        :param strip_last_seconds: If True, this method will return minute-long segments,
                                   but the last three seconds of this AudioSegment won't be returned.
                                   This is useful for removing the microphone artifact at the end of the recording.
        :returns: A list of AudioSegment objects, each of which is one minute long at most
                  (and only the last one - if any - will be less than one minute).
        """
        outs = self.dice(seconds=60, zero_pad=False)

        # Now cut out the last three seconds of the last item in outs (it will just be microphone artifact)
        # or, if the last item is less than three seconds, just get rid of it
        if strip_last_seconds:
            if outs[-1].duration_seconds > 3:
                outs[-1] = outs[-1][:-MS_PER_S * 3]
            else:
                outs = outs[:-1]

        return outs

    def zero_extend(self, duration_s=None, num_samples=None):
        """
        Adds a number of zeros (digital silence) to the AudioSegment (returning a new one).

        :param duration_s: The number of seconds of zeros to add. If this is specified, `num_samples` must be None.
        :param num_samples: The number of zeros to add. If this is specified, `duration_s` must be None.
        :returns: A new AudioSegment object that has been zero extended.
        :raises: ValueError if duration_s and num_samples are both specified.
        """
        if duration_s is not None and num_samples is not None:
            raise ValueError("`duration_s` and `num_samples` cannot both be specified.")
        elif duration_s is not None:
            num_samples = self.frame_rate * duration_s
        seg = AudioSegment(self.seg, self.name)
        zeros = silent(duration=num_samples / self.frame_rate, frame_rate=self.frame_rate)
        return zeros.overlay(seg)

def deserialize(bstr):
    """
    Attempts to deserialize a bytestring into an audiosegment.

    :param bstr: The bytestring serialized via an audiosegment's serialize() method.
    :returns: An AudioSegment object deserialized from `bstr`.
    """
    d = pickle.loads(bstr)
    seg = pickle.loads(d['seg'])
    return AudioSegment(seg, d['name'])

def empty():
    """
    Creates a zero-duration AudioSegment object.

    :returns: An empty AudioSegment object.
    """
    dubseg = pydub.AudioSegment.empty()
    return AudioSegment(dubseg, "")

def from_file(path):
    """
    Returns an AudioSegment object from the given file based on its file extension.
    If the extension is wrong, this will throw some sort of error.

    :param path: The path to the file, including the file extension.
    :returns: An AudioSegment instance from the file.
    """
    _name, ext = os.path.splitext(path)
    ext = ext.lower()[1:]
    seg = pydub.AudioSegment.from_file(path, ext)
    return AudioSegment(seg, path)

def from_mono_audiosegments(*args):
    """
    Creates a multi-channel AudioSegment out of multiple mono AudioSegments (two or more). Each mono
    AudioSegment passed in should be exactly the same number of samples.

    :returns: An AudioSegment of multiple channels formed from the given mono AudioSegments.
    """
    return AudioSegment(pydub.AudioSegment.from_mono_audiosegments(*args), "")

def from_numpy_array(nparr, framerate):
    """
    Returns an AudioSegment created from the given numpy array.

    The numpy array must have shape = (num_samples, num_channels).

    :param nparr: The numpy array to create an AudioSegment from.
    :returns: An AudioSegment created from the given array.
    """
    # interleave the audio across all channels and collapse
    if nparr.dtype.itemsize not in (1, 2, 4):
        raise ValueError("Numpy Array must contain 8, 16, or 32 bit values.")
    if len(nparr.shape) == 1:
        arrays = [nparr]
    elif len(nparr.shape) == 2:
        arrays = [nparr[i,:] for i in range(nparr.shape[0])]
    else:
        raise ValueError("Numpy Array must be one or two dimensional. Shape must be: (num_samples, num_channels).")
    interleaved = np.vstack(arrays).reshape((-1,), order='F')
    dubseg = pydub.AudioSegment(interleaved.tobytes(),
                                frame_rate=framerate,
                                sample_width=interleaved.dtype.itemsize,
                                channels=len(interleaved.shape)
                               )
    return AudioSegment(dubseg, "")

def silent(duration=1000, frame_rate=11025):
    """
    Creates an AudioSegment object of the specified duration/frame_rate filled with digital silence.

    :param duration: The duration of the returned object in ms.
    :param frame_rate: The samples per second of the returned object.
    :returns: AudioSegment object filled with pure digital silence.
    """
    seg = pydub.AudioSegment.silent(duration=duration, frame_rate=frame_rate)
    return AudioSegment(seg, "")

