from utility.audio import AudioPreprocessing
from optparse import OptionParser

import sys


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-v', '--video_path', dest='path', default='')
    parser.add_option('-d', '--output_dir', dest='dir', default='')

    (options, args) = parser.parse_args(sys.argv)

    path = options.path
    out_dir = options.dir

    audioprocessing = AudioPreprocessing(sr=16000)
    audioprocessing.extract_audio_track(path, out_dir)
