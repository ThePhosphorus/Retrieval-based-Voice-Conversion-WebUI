from pathlib import Path
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, audio
from neutone_sdk.utils import load_neutone_model
import torch

wrapper, metadata = load_neutone_model("./model.nm")

input_sample = audio.AudioSample.from_file(Path(".") / "samples" / "Max_Original.wav")
rendered_sample = audio.render_audio_sample(wrapper, input_sample)
sample_pair =  audio.AudioSamplePair(input_sample, rendered_sample)
metadata = sample_pair.to_metadata_format()
with open(Path(".") / "samples" / f"Max_Modified_full6.mp3", "wb") as f:
           f.write(rendered_sample.to_mp3_bytes())

