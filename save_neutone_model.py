from typing import Dict, List, Optional
import torch
from torch import nn
from torch.nn import functional as F
from neutone_sdk import WaveformToWaveformBase, NeutoneParameter, audio
import neutone_sdk.audio as audio
from infer.lib.infer_pack.models import SynthesizerTrnMs768NSFsid
import torchaudio
import torchcrepe
# from infer.lib.rmvpe import RMVPE
import math
from scipy import signal
import numpy as np

from fairseq import checkpoint_utils
from fairseq.models import hubert


class CharleneModel(torch.nn.Module):
    def __init__(self):
        super(CharleneModel, self).__init__()  # Call the superclass constructor

        # self.model
        self.device = "cpu"
        self.vec_channels = 768
        self.is_half = False
        self.sr = 16_000
        self.window = 160

        self.cpt = torch.load("assets/weights/charlene_blanchette_modified.pth", map_location="cpu")
        self.cpt["config"][-3] = self.cpt["weight"]["emb_g.weight"].shape[0]
        self.net_g = SynthesizerTrnMs768NSFsid(*self.cpt["config"], is_half=False)
        self.net_g.load_state_dict(self.cpt["weight"], strict=False)

        self.hubert_model = torchaudio.models.hubert_base()
        self.hubert_model.load_state_dict(torch.load('assets/hubert/hubert_base_torch.pt'))

        self.crepe = torchcrepe.TorchCrepe('full', self.device)
        self.bh, self.ah = signal.butter(N=5, Wn=48, btype="high", fs=16_000)
        self.bh = torch.from_numpy(self.bh)
        self.ah = torch.from_numpy(self.ah)

        self.t_pad = 16_000 //2
        self.t_pad_tgt = 40_000 //2


    def get_pitch(self, phone, transpose: int):
        ###
        f0_min = 50
        f0_max = 1100
        f0_mel_min = 1127 * math.log(1 + f0_min / 700)
        f0_mel_max = 1127 * math.log(1 + f0_max / 700)

        # Pick a batch size that doesn't cause memory errors on your gpu
        batch_size = 512
        # Compute pitch using first gpu
        audio = phone
        f0, pd =  self.predict(
            audio=audio, 
            fmin=float(f0_min), 
            fmax=float(f0_max),  
            batch_size=batch_size)
        pd = torchcrepe.filter.median(pd, 3)
        f0 = torchcrepe.filter.mean(f0, 3)
        f0[pd < 0.1] = 0
        ###

        f0 *= pow(2, transpose / 12)
        f0bak = f0.clone()
        f0_mel = 1127 * torch.log(1 + f0 / 700)
        f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * 254 / (
            f0_mel_max - f0_mel_min
        ) + 1
        f0_mel[f0_mel <= 1] = 1
        f0_mel[f0_mel > 255] = 255
        f0_coarse = torch.round(f0_mel).to(torch.int32)
        return f0_coarse, f0bak  # 1-0

    def predict(self, audio: torch.Tensor,
            fmin: float=50.,
            fmax: float=2006.0,
            batch_size: Optional[int]=None,
            pad: bool=True) :
        pitch_result: List[torch.Tensor] = []
        periodicity_result: List[torch.Tensor] = []
        PITCH_BINS = 360

        # Postprocessing breaks gradients, so just don't compute them
        with torch.no_grad():

            # Preprocess audio
            generator = torchcrepe.preprocess(audio,
                                self.sr,
                                self.window,
                                batch_size,
                                self.device,
                                pad)
            for frames in generator:

                # Infer independent probabilities for each pitch bin
                probabilities = self.crepe(frames, embed=False)

                # shape=(batch, 360, time / hop_length)
                probabilities = probabilities.reshape(
                    audio.size(0), -1, PITCH_BINS).transpose(1, 2)

                # Convert probabilities to F0 and periodicity
                pitch, periodicity = torchcrepe.postprocess(probabilities,
                                    fmin,
                                    fmax)

                # Place on same device as audio to allow very long inputs
                pitch_result.append(pitch.to(audio.device))
                periodicity_result.append(periodicity.to(audio.device))
        return torch.cat(pitch_result, 1), torch.cat(periodicity_result, 1)


    def forward(self, phone: torch.Tensor, pitch: int):

        input_shape = phone.shape[1] - (self.t_pad)
        phone = phone.to(dtype=torch.float64)
        phone = torchaudio.functional.filtfilt(phone[0], self.ah, self.bh)[None]


        audio_pad = phone.float()
        #audio_pad = F.pad(phone, (0, self.t_pad), mode="reflect").float()

        p_len = audio_pad.shape[1] // self.window

        features, _ = self.hubert_model(audio_pad.to(self.device))

        features = F.interpolate(features.permute(0, 2, 1), scale_factor=2.0).permute(0, 2, 1)

        if features.shape[1] < p_len:
            p_len = features.shape[1]

        pitch, pitchf = self.get_pitch(audio_pad, pitch)
        
        pitch = pitch[:, :p_len]
        pitchf = pitchf[:, :p_len]
        
        phone_lengths = torch.tensor([p_len]).long().to(self.device)  # hidden unit length (seems useless)
        sid = torch.LongTensor([0]).to(self.device)  # Speaker ID
        
        output = self.net_g.infer(features, phone_lengths, pitch, pitchf, sid)
        output = output[0][0, 0].data.float()

        # pad_size = (output.shape[0] - (input_shape * 40 // 16) ) - self.t_pad_tgt

        # output = (output[self.t_pad_tgt : -pad_size] if pad_size > 0 else output  )[None]
        output = output[None]

        output = torchaudio.functional.resample(output, orig_freq=40000, new_freq=16000)
        return output

class CharleneModelWrapper(WaveformToWaveformBase):
    def setup(self, buffer_size: int) :
        self.buffer_size = int(buffer_size)
        self.overlap_n_samples = self.model.t_pad
        self.model_segment_size = self.buffer_size + self.overlap_n_samples
        self.fade_up = torch.linspace(0,1, max(self.overlap_n_samples, 1))
        self.fade_down = torch.linspace(1,0, max(self.overlap_n_samples, 1))
        self.in_buf = torch.zeros(1 if self.is_input_mono() else 2, self.model_segment_size)
        self.in_buf_tmp = torch.zeros(1 if self.is_input_mono() else 2, self.model_segment_size)
        self.out_buf = torch.zeros(1 if self.is_input_mono() else 2, self.model_segment_size)

    def get_model_name(self) -> str:
        return "circonflex"

    def get_model_authors(self) -> List[str]:
        return ["Jonathan Girard", "Adem Aber Aouni"]

    def get_model_short_description(self) -> str:
        return "Vocal Transformer using RVC trained model."

    def get_model_long_description(self) -> str:
        return "Vocal Transformer using RVC trained model."

    def get_technical_description(self) -> str:
        return "Vocal Transformer using RVC trained model."

    def get_technical_links(self) -> Dict[str, str]:
        return {
            "Code": "https://github.com/QosmoInc/neutone_sdk/blob/main/examples/example_clipper.py"
        }

    def get_tags(self) -> List[str]:
        return ["RVC"]

    def get_model_version(self) -> str:
        return "1.0.0"

    def is_experimental(self) -> bool:
        return True

    def get_neutone_parameters(self) -> List[NeutoneParameter]:
        return [
            NeutoneParameter("pitch", "pitch", default_value=0.5),
            # NeutoneParameter("max", "max clip threshold", default_value=0.15),
            # NeutoneParameter("gain", "scale clip threshold", default_value=1.0),
        ]

    @torch.jit.export
    def is_input_mono(self) -> bool:
        return True

    @torch.jit.export
    def is_output_mono(self) -> bool:
        return True

    @torch.jit.export
    def get_native_sample_rates(self) -> List[int]:
        return [ 16000 ]  # Supports all sample rates
    
    # @torch.jit.export
    # def get_look_behind_samples(self) -> int:
    #     return self.model.t_pad

    @torch.jit.export
    def get_native_buffer_sizes(self) -> List[int]:
        return [16_000]  # Supports all buffer sizes
    
    def set_model_sample_rate_and_buffer_size(self, sample_rate: int, n_samples: int) -> bool:
        self.setup(n_samples)
        return True
    
    @torch.jit.export
    def calc_min_delay_samples(self) -> int:
        return self.overlap_n_samples
    
    @torch.jit.export
    def reset(self) -> None:
        self.in_buf.fill_(0)
        self.in_buf_tmp.fill_(0)
        self.out_buf.fill_(0)

    #def aggregate_params(self, params: torch.Tensor) -> torch.Tensor:
    #    return params  # We want sample-level control, so no aggregation

    @torch.no_grad()
    def do_forward_pass(self, x: torch.Tensor, params: Dict[str, torch.Tensor]) -> torch.Tensor:
        # Rotate previous input buffer to the left
        self.in_buf[:, : self.overlap_n_samples] = self.in_buf[:, -self.overlap_n_samples:]
        self.in_buf[:, -self.buffer_size:] = x

        float_pitch = (params["pitch"].item() - 0.5) * 12
        pitch = int(float_pitch)

        out = self.model.forward(self.in_buf, pitch)

        # apply faders to the output buffer
        self.out_buf[:, -self.overlap_n_samples:] *= self.fade_down
        out[:, :self.overlap_n_samples] *= self.fade_up
        out[:, :self.overlap_n_samples] += self.out_buf[:, -self.overlap_n_samples:]
        self.out_buf = out

        return out[:, :self.buffer_size]
    
from pathlib import Path
from neutone_sdk.utils import save_neutone_model, dump_samples_from_metadata
import torch

class DataHolder(torch.nn.Module):
    def __init__(self) :
        super(DataHolder, self).__init__()  # Call the superclass constructor
        cpt = torch.load("assets/weights/charlene_blanchette_modified.pth", map_location="cpu")
        cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
        self.values = cpt

    def forward(x) :
        return x

model = CharleneModel()
wrapper = CharleneModelWrapper(model)

# input_sample = audio.AudioSample.from_file(Path(".") / "samples" / "Max_Original.wav")
# rendered_sample = audio.render_audio_sample(wrapper, input_sample)
# sample_pair =  audio.AudioSamplePair(input_sample, rendered_sample)
# metadata = sample_pair.to_metadata_format()
# with open(Path(".") / "samples" / f"Max_Modified_full6.mp3", "wb") as f:
#            f.write(rendered_sample.to_mp3_bytes())

# audio = torch.from_numpy(np.load("./original.npy"))[None]

# result = wrapper(audio)

# torchaudio.save('./samples/result_2.mp3', result, sample_rate= 16_000)


save_neutone_model(wrapper, Path.cwd(), dump_samples=True)