from essentia.standard import MonoLoader,Windowing,Spectrum,RMS,Centroid,FrequencyBands,FrameGenerator,CartesianToPolar
import madmom.audio.signal as signal
from sklearn import preprocessing
from essentia import array
import numpy as np
from madmom.features.onsets import SpectralOnsetProcessor, OnsetPeakPickingProcessor
from madmom.audio.filters import LogarithmicFilterbank

def rms_centroids(filename, frameSize = 1024, hopSize = 512, sampleRate=44100):
    # load our audio into an array
    audio = MonoLoader(filename=filename, sampleRate=44100)()

    # create the pool and the necessary algorithms
    w = Windowing()
    spec = Spectrum()
    rms = RMS()
    centroid = Centroid(range=int(sampleRate/2))
    cs = []
    rmss = []
    # compute the centroid for all frames in our audio and add it to the pool
    for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
        sf = spec(w(frame))
        cs.append(centroid(sf))
        rmss.append(rms(sf))
    return np.array(rmss), np.array(cs)

def combine_series(events, delta):
    """
    Combine all events of series with inner distances
    less then delta.


    Parameters
    ----------
    events : list or numpy array
        Events to be combined.
    delta : float
        Combination delta.
    Returns
    -------
    numpy array
        Combined events.

    """
    # add a small value to delta, otherwise we end up in floating point hell
    delta += 1e-12
    # return immediately if possible
    if len(events) <= 1:
        return events
    # create working copy
    events = np.array(events, copy=True)
    # set start position
    idx = 0
    # iterate over all remaining events
    for right in events[1:]:
        if right - events[idx] > delta:
            idx += 1
        events[idx] = right
    return events[:idx + 1]
    
def GuitarOnsetDetector(audio_filename, fs = 44100):
    # TODO: ad hoc!!!! Derive it automatically.
    onset_threshold = 2
    series_delta=0.22
    fps=180
    fs = 44100
    hopSize = int(fs/fps)
    max_spectral_centroid = 3500
    # fps must be a divisor of fs to obtain integer hopSize
    # (it just simplifies the code below)
    sodf = SpectralOnsetProcessor(onset_method='superflux', fps=fps,
                                  filterbank=LogarithmicFilterbank,
                                  num_bands=24, log=np.log10)
    sodf_onsets = sodf(audio_filename)
    # "fusion" with rms-diff.
    rms, cs = rms_centroids(audio_filename, frameSize=1024, hopSize=hopSize, sampleRate=fs)
    rms = signal.smooth(rms, int(fs / hopSize * 0.2))
    rms = preprocessing.scale(rms, with_mean=False, copy=False)
    rms = rms[1:] - rms[:-1]

    sodf_onsets[rms <= 0] = 0
    #sodf_onsets = sodf_onsets * np.power(rms, 0.01)
    #sodf_onsets[np.isnan(sodf_onsets)] = 0

    proc = OnsetPeakPickingProcessor(
        fps=fps, threshold=onset_threshold)
    p_onsets = proc(sodf_onsets)
    p_onsets = combine_series(p_onsets,series_delta)
    smoothed = []
    for i in range(len(p_onsets)):
        onset = p_onsets[i]
        duration = 0.5
        if (i < len(p_onsets) - 1):
            duration = min((p_onsets[i + 1] - p_onsets[i]), duration)
        window_len = int(duration * fs / hopSize)
        s = int(float(onset) * fs / hopSize)
        d = min(window_len, len(cs) - s)
        w = eval('np.hanning(2*d)')
        w = w[d:] / np.sum(w[d:])
        w = np.reshape(w, (1, d))
        c = cs[s:s + d]
        smoothed.append(np.dot(w, c)[0])
    result = []
    for i in range(len(p_onsets)):
        if smoothed[i] < max_spectral_centroid:
            result.append(p_onsets[i])
    return result