from spleeter.separator import Separator
import mirdata
import os
# Using embedded configuration.
separator = Separator('spleeter:4stems-16kHz')
print('Spleeter Loaded')

gtzan = mirdata.initialize('gtzan_genre')
gtzan.download()
print(len(gtzan.track_ids))

track_ids = gtzan.track_ids
print(gtzan.track(track_ids[42]))

from spleeter.audio.adapter import AudioAdapter
from tqdm import tqdm
import librosa
import numpy as np

audio_loader = AudioAdapter.default()
sample_rate = 16_000
extract_keys = ['drums', 'other', 'vocals']

data_path = "/raid/home/niranjan20090/DL/gtzan_full"

for track_id in tqdm(track_ids):
    track_wavs = []
    waveform, _ = audio_loader.load(gtzan.track(track_id).audio_path, sample_rate=16_000)
    waveforms = separator.separate(waveform)
    track_wavs.append(waveform.squeeze())
    for key in extract_keys:
        track_wavs.append(librosa.to_mono(waveforms[key].T).squeeze())
    track_wavs = np.stack(track_wavs)
    np.save(os.path.join(data_path,f'{track_id}.npy'),track_wavs)