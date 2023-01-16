import numpy as np

def load_chroma_features():
    x_short = np.load('chroma_npy/StairwayToHeaven_MakesMeWonder_chroma.npy')
    x_long  = np.load('chroma_npy/StairwayToHeaven_1min_chroma.npy')

    return x_short, x_long

def load_wav():
    x_short = np.load('wav_npy/StairwayToHeaven_MakesMeWonder.npy')
    x_long  = np.load('wav_npy/StairwayToHeaven_1min.npy')

    return x_short, x_long

def get_subset_wav(index_end):
    x_short_chroma, x_long_chroma = load_chroma_features()
    x_short_wav, x_long_wav = load_wav()
    ratio_chroma_wav = len(x_long_wav) // len(x_long_chroma)

    index_end_wav = index_end * ratio_chroma_wav
    index_start_wav = max(0, index_end_wav - len(x_short_wav))

    return x_long_wav[index_start_wav:index_end_wav]

