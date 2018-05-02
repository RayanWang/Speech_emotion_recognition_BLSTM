from pydub import AudioSegment
from array import array
from struct import pack

import os
import glob
import audiosegment
import webrtcvad
import sys
import wave
import time


RATE = 16000
CHUNK_DURATION_MS = 20  # supports 10, 20 (ms)
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS * 2 / 1000)  # chunk to read in bytes
NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400ms / 20ms  frame
NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2

vad = webrtcvad.Vad(mode=3)


def extract_audio_track(videopath, audiopath):
    for video in glob.glob(videopath + '*.mp4'):
        wav_filename = audiopath + os.path.splitext(os.path.basename(video))[0] + '.wav'
        AudioSegment.from_file(video).export(wav_filename, format='wav')


def record_to_file(path, data, sample_width):
    '''
    Records from the wav audio and outputs the resulting data to 'path'
    '''
    data = pack('<' + ('h' * len(data)), *data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def normalize(snd_data):
    '''
    Average the volume out
    '''
    max_value = max(abs(i) for i in snd_data)
    if max_value == 0:
        return snd_data
    maximum = 32767
    times = float(maximum) / max_value
    r = array('h')
    for i in snd_data:
        r.append(int(i * times))
    return r


def voice_segmentation(filename, outdir):
    print("Re-sampling...")
    seg = audiosegment.from_file(filename).resample(sample_rate_Hz=16000, sample_width=2, channels=1)

    print("Detecting voice...")

    got_a_sentence = False
    ended = False
    offset = CHUNK_DURATION_MS

    path = filename.split('/')[-1]
    path = outdir + '/' + path

    i = 1
    while not ended:
        triggered = False
        ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
        ring_buffer_index = 0

        ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
        ring_buffer_index_end = 0

        raw_data = array('h')
        index = 0
        start_point = 0
        start_time = time.time()

        while not got_a_sentence and not ended:
            chunk = seg[(offset - CHUNK_DURATION_MS):offset].raw_data

            raw_data.extend(array('h', chunk))
            offset += CHUNK_DURATION_MS
            index += CHUNK_SIZE
            time_used = time.time() - start_time

            active = vad.is_speech(chunk, RATE)

            ring_buffer_flags[ring_buffer_index] = 1 if active else 0
            ring_buffer_index += 1
            ring_buffer_index %= NUM_WINDOW_CHUNKS

            ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
            ring_buffer_index_end += 1
            ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

            # start point detection
            if not triggered:
                num_voiced = sum(ring_buffer_flags)
                if num_voiced > 0.8 * NUM_WINDOW_CHUNKS:
                    sys.stdout.write(' Start sentence ')
                    triggered = True
                    start_point = index - CHUNK_SIZE * 20  # start point
            # end point detection
            else:
                num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
                if num_unvoiced > 0.90 * NUM_WINDOW_CHUNKS_END or time_used > 10:
                    sys.stdout.write(' End sentence ')
                    triggered = False
                    got_a_sentence = True

            if offset >= len(seg):
                print('File end')
                ended = True

            sys.stdout.flush()

        sys.stdout.write('\n')

        got_a_sentence = False

        data_size = len(raw_data.tolist())
        if (data_size <= start_point) or (start_point < 0):
            print('No data to write')
            continue

        print('Start point: %d' % start_point)

        # write to file
        raw_data.reverse()
        for _ in range(start_point):
            raw_data.pop()

        data_size = len(raw_data.tolist())
        if data_size:
            raw_data.reverse()
            raw_data = normalize(raw_data)

            f, ext = os.path.splitext(path)
            f = f + '_' + str(i) + ext
            record_to_file(f, raw_data, 2)

            i += 1
        else:
            print("Data is empty, it's not make sense.")
