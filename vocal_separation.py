from utility.audio import AudioSplitter
from optparse import OptionParser

import glob
import sys

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-p', '--wav_path', dest='path', default='')
    parser.add_option('-d', '--output_dir', dest='dir', default='')

    (options, args) = parser.parse_args(sys.argv)

    path = options.path
    out_dir = options.dir

    splitter = AudioSplitter(sr=16000)
    for wav in glob.glob(path + '/*.wav'):
        splitter.split_vocal_to_wav(wav, out_dir)
