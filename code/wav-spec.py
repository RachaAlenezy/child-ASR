from pathlib import Path
import matplotlib.pyplot as plot
from scipy.io import wavfile





def main():
    # if len(args) != 2:
    #     sys.stderr.write(
    #         'Usage: silenceremove.py <aggressiveness> <path to wav file>\n')
    #     sys.exit(1)
    path = Path('data').glob('**/*.wav')
    wavs = [str(wavf) for wavf in path if wavf.is_file()]
    wavs.sort()
    print(wavs[0])

    number_of_files=len(wavs)

    spk_ID = [wavs[i].split('/')[-1].lower() for i in range(number_of_files)]

    for i in range(number_of_files):
        samplingfrequency, signaldata = wavfile.read(wavs[i])
        pxx, freq, bins, im = plot.specgram(x=signaldata, Fs=samplingfrequency, noverlap=384, NFFT=512)
        plot.title('spec of vowel')
        plot.xlabel('time')
        plot.ylabel('freq')
        plot.savefig("new_specks/spk_ID:{}.png".format(spk_ID[i]), bbox_inches='tight', dpi=300, frameon='false')

if __name__ == '__main__':
    main()
