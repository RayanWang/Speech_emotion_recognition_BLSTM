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

    audioprocessing = AudioPreprocessing(video_path=path, out_path=out_dir)
    audioprocessing.extract_audio_track()
