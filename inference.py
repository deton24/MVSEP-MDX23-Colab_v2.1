# coding: utf-8

if __name__ == '__main__':
    import os
     
    gpu_use = "0"

    print('GPU use: {}'.format(gpu_use))
    os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(gpu_use)
import warnings
warnings.filterwarnings("ignore")

import inspect
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import soundfile as sf
from demucs.states import load_model
from demucs import pretrained
from demucs.apply import apply_model
import onnxruntime as ort
from time import time
import librosa
import hashlib
from scipy import signal
import gc
import yaml
from ml_collections import ConfigDict
import sys
import math
import pathlib
import warnings
from scipy.signal import resample_poly

from modules.tfc_tdf_v2 import Conv_TDF_net_trim_model
from modules.tfc_tdf_v3 import TFC_TDF_net, STFT
from modules.segm_models import Segm_Models_Net
from modules.bs_roformer import BSRoformer
from modules.bs_roformer import MelBandRoformer
from modules.scnet import SCNet



def get_models(name, device, load=True, vocals_model_type=0):
    if vocals_model_type == 2:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='vocals',
            L=11,
            n_fft=7680,
            dim_f=3072
        )
    elif vocals_model_type == 3:
        model_vocals = Conv_TDF_net_trim_model(
            device=device,
            target_name='instrum',
            L=11,
            n_fft=5120,
            dim_f=2560
        )

    return [model_vocals]


def get_model_from_config(model_type, config_path):
    with open(config_path) as f:
        config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))
        if model_type == 'mdx23c':
            # from modules.tfc_tdf_v3 import TFC_TDF_net
            model = TFC_TDF_net(config)
        elif model_type == 'segm_models':
            # from modules.segm_models import Segm_Models_Net
            model = Segm_Models_Net(config)
        elif model_type == 'bs_roformer':
            # from modules.bs_roformer import BSRoformer
            model = BSRoformer(
                **dict(config.model)
            )
        elif model_type == 'mel_band_roformer':
            # from modules.mel_band_roformer import MelBandRoformer
            model = MelBandRoformer(
                **dict(config.model)
            )
        else:
            print('Unknown model: {}'.format(model_type))
            model = None
    return model, config


def demix_new(model, mix, device, config, dim_t=256):
    mix = torch.tensor(mix)
    #N = options["overlap_BSRoformer"]
    N = 2 # overlap 50%
    batch_size = 1
    mdx_window_size = dim_t
    C = config.audio.hop_length * (mdx_window_size - 1)
    fade_size = C // 100
    step = int(C // N)
    border = C - step
    length_init = mix.shape[-1]
    #print(f"1: {mix.shape}")
    
    # Do pad from the beginning and end to account floating window results better
    if length_init > 2 * border and (border > 0):
        mix = nn.functional.pad(mix, (border, border), mode='reflect')
        
    
    # Prepare windows arrays (do 1 time for speed up). This trick repairs click problems on the edges of segment
    window_size = C
    fadein = torch.linspace(0, 1, fade_size)
    fadeout = torch.linspace(1, 0, fade_size)
    window_start = torch.ones(window_size)
    window_middle = torch.ones(window_size)
    window_finish = torch.ones(window_size)
    window_start[-fade_size:] *= fadeout # First audio chunk, no fadein
    window_finish[:fade_size] *= fadein # Last audio chunk, no fadeout
    window_middle[-fade_size:] *= fadeout
    window_middle[:fade_size] *= fadein




    with torch.cuda.amp.autocast():
        with torch.inference_mode():
            if config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(config.training.instruments),) + tuple(mix.shape)

            result = torch.zeros(req_shape, dtype=torch.float32)
            counter = torch.zeros(req_shape, dtype=torch.float32)
            i = 0
            batch_data = []
            batch_locations = []
            while i < mix.shape[1]:
                # print(i, i + C, mix.shape[1])
                part = mix[:, i:i + C].to(device)
                length = part.shape[-1]
                if length < C:
                    if length > C // 2 + 1:
                        part = nn.functional.pad(input=part, pad=(0, C - length), mode='reflect')
                    else:
                        part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                batch_data.append(part)
                batch_locations.append((i, length))
                i += step

                if len(batch_data) >= batch_size or (i >= mix.shape[1]):
                    arr = torch.stack(batch_data, dim=0)
                    x = model(arr)

                    window = window_middle
                    if i - step == 0:  # First audio chunk, no fadein
                        window = window_start
                    elif i >= mix.shape[1]:  # Last audio chunk, no fadeout
                        window = window_finish

                    for j in range(len(batch_locations)):
                        start, l = batch_locations[j]
                        result[..., start:start+l] += x[j][..., :l].cpu() * window[..., :l]
                        counter[..., start:start+l] += window[..., :l]

                    batch_data = []
                    batch_locations = []

            estimated_sources = result / counter
            estimated_sources = estimated_sources.cpu().numpy()
            np.nan_to_num(estimated_sources, copy=False, nan=0.0)

            if length_init > 2 * border and (border > 0):
                # Remove pad
                estimated_sources = estimated_sources[..., border:-border]

    if config.training.target_instrument is None:
        return {k: v for k, v in zip(config.training.instruments, estimated_sources)}
    else:
        return {k: v for k, v in zip([config.training.target_instrument], estimated_sources)}


def demix_new_wrapper(mix, device, model, config, dim_t=256, bigshifts=1):
    if bigshifts <= 0:
        bigshifts = 1

    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results = []

    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix_new(model, shifted_mix, device, config, dim_t=dim_t)
        vocals = next(sources[key] for key in sources.keys() if key.lower() == "vocals")
        unshifted_vocals = np.concatenate((vocals[..., shift:], vocals[..., :shift]), axis=-1)  
        vocals *= 1 # 1.0005168 CHECK NEEDED! volume compensation
        
        results.append(unshifted_vocals)

    vocals = np.mean(results, axis=0)
    
    return vocals

def demix_vitlarge(model, mix, device):
    C = model.config.audio.hop_length * (2 * model.config.inference.dim_t - 1)
    N = 2
    step = C // N

    with torch.cuda.amp.autocast():
        with torch.no_grad():
            if model.config.training.target_instrument is not None:
                req_shape = (1, ) + tuple(mix.shape)
            else:
                req_shape = (len(model.config.training.instruments),) + tuple(mix.shape)

            mix = mix.to(device)
            result = torch.zeros(req_shape, dtype=torch.float32).to(device)
            counter = torch.zeros(req_shape, dtype=torch.float32).to(device)
            i = 0

            while i < mix.shape[1]:
                part = mix[:, i:i + C]
                length = part.shape[-1]
                if length < C:
                    part = nn.functional.pad(input=part, pad=(0, C - length, 0, 0), mode='constant', value=0)
                x = model(part.unsqueeze(0))[0]
                result[..., i:i+length] += x[..., :length]
                counter[..., i:i+length] += 1.
                i += step
            estimated_sources = result / counter

    if model.config.training.target_instrument is None:
        return {k: v for k, v in zip(model.config.training.instruments, estimated_sources.cpu().numpy())}
    else:
        return {k: v for k, v in zip([model.config.training.target_instrument], estimated_sources.cpu().numpy())}


def demix_full_vitlarge(mix, device, model):
    if options["BigShifts"] <= 0:
        bigshifts = 1
    else:
        bigshifts = options["BigShifts"]
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]

    results1 = []
    results2 = []
    mix = torch.from_numpy(mix).type('torch.FloatTensor').to(device)
    for shift in tqdm(shifts, position=0):
        shifted_mix = torch.cat((mix[:, -shift:], mix[:, :-shift]), dim=-1)
        sources = demix_vitlarge(model, shifted_mix, device)
        sources1 = sources["vocals"]
        sources2 = sources["other"]
        restored_sources1 = np.concatenate((sources1[..., shift:], sources1[..., :shift]), axis=-1)
        restored_sources2 = np.concatenate((sources2[..., shift:], sources2[..., :shift]), axis=-1)
        results1.append(restored_sources1)
        results2.append(restored_sources2)


    sources1 = np.mean(results1, axis=0)
    sources2 = np.mean(results2, axis=0)

    return sources1, sources2


def demix_wrapper(mix, device, models, infer_session, overlap=0.2, bigshifts=1, vc=1.0):
    if bigshifts <= 0:
        bigshifts = 1
    shift_in_samples = mix.shape[1] // bigshifts
    shifts = [x * shift_in_samples for x in range(bigshifts)]
    results = []
    
    for shift in tqdm(shifts, position=0):
        shifted_mix = np.concatenate((mix[:, -shift:], mix[:, :-shift]), axis=-1)
        sources = demix(shifted_mix, device, models, infer_session, overlap) * vc # 1.021 volume compensation
        restored_sources = np.concatenate((sources[..., shift:], sources[..., :shift]), axis=-1)
        results.append(restored_sources)
        
    sources = np.mean(results, axis=0)
    
    return sources

def demix(mix, device, models, infer_session, overlap=0.2):
    start_time = time()
    sources = []
    n_sample = mix.shape[1]
    n_fft = models[0].n_fft
    n_bins = n_fft//2+1
    trim = n_fft//2
    hop = models[0].hop
    dim_f = models[0].dim_f
    dim_t = models[0].dim_t # * 2
    chunk_size = hop * (dim_t -1)
    org_mix = mix
    tar_waves_ = []
    mdx_batch_size = 1
    overlap = overlap
    gen_size = chunk_size-2*trim
    pad = gen_size + trim - ((mix.shape[-1]) % gen_size)
    
    mixture = np.concatenate((np.zeros((2, trim), dtype='float32'), mix, np.zeros((2, pad), dtype='float32')), 1)

    step = int((1 - overlap) * chunk_size)
    result = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    divider = np.zeros((1, 2, mixture.shape[-1]), dtype=np.float32)
    total = 0
    total_chunks = (mixture.shape[-1] + step - 1) // step

    for i in range(0, mixture.shape[-1], step):
        total += 1
        start = i
        end = min(i + chunk_size, mixture.shape[-1])
        chunk_size_actual = end - start

        if overlap == 0:
            window = None
        else:
            window = np.hanning(chunk_size_actual)
            window = np.tile(window[None, None, :], (1, 2, 1))

        mix_part_ = mixture[:, start:end]
        if end != i + chunk_size:
            pad_size = (i + chunk_size) - end
            mix_part_ = np.concatenate((mix_part_, np.zeros((2, pad_size), dtype='float32')), axis=-1)
        
        
        mix_part = torch.tensor([mix_part_], dtype=torch.float32).to(device)
        mix_waves = mix_part.split(mdx_batch_size)
        
        with torch.no_grad():
            for mix_wave in mix_waves:
                _ort = infer_session
                stft_res = models[0].stft(mix_wave)
                stft_res[:, :, :3, :] *= 0 
                res = _ort.run(None, {'input': stft_res.cpu().numpy()})[0]
                ten = torch.tensor(res)
                tar_waves = models[0].istft(ten.to(device))
                tar_waves = tar_waves.cpu().detach().numpy()
                
                if window is not None:
                    tar_waves[..., :chunk_size_actual] *= window 
                    divider[..., start:end] += window
                else:
                    divider[..., start:end] += 1
                result[..., start:end] += tar_waves[..., :end-start]


    tar_waves = result / divider
    tar_waves_.append(tar_waves)
    tar_waves_ = np.vstack(tar_waves_)[:, :, trim:-trim]
    tar_waves = np.concatenate(tar_waves_, axis=-1)[:, :mix.shape[-1]]
    source = tar_waves[:,0:None]

    return source
class EnsembleDemucsMDXMusicSeparationModel:
    """
    Doesn't do any separation just passes the input back as output
    """
    def __init__(self, options):
        """
            options - user options
        """
        # Device setup
        if torch.cuda.is_available() and not options.get('cpu', False):
            self.device = 'cuda:0'
        else:
            self.device = 'cpu'
        
        self.single_onnx = options.get('single_onnx', False)
        self.overlap_demucs = min(max(float(options['overlap_demucs']), 0.0), 0.99)
        self.overlap_MDX = min(max(float(options['overlap_VOCFT']), 0.0), 0.99)
        
        # Model folder
        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
        
        self.options = options


        # Execution providers for ONNX
        if self.device == 'cpu':
            self.providers = ["CPUExecutionProvider"]
        else:
            self.providers = ["CUDAExecutionProvider"]
        
        # Preloading ensemble models (if not vocals only)
        if not options.get('vocals_only', False):
            self.models = []
            self.weights_vocals = np.array([10, 1, 8, 9])
            self.weights_bass = np.array([19, 4, 5, 8])
            self.weights_drums = np.array([18, 2, 4, 9])
            self.weights_other = np.array([14, 2, 5, 10])
            
            model_names = ['htdemucs_ft', 'htdemucs', 'htdemucs_6s', 'hdemucs_mmi']
            for model_name in model_names:
                model = pretrained.get_model(model_name)
                model.to(self.device)
                self.models.append(model)

    def download_file_if_not_exists(self, remote_url, local_path):
        """Downloads a file from a URL if it does not already exist."""
        if not os.path.isfile(local_path):
            torch.hub.download_url_to_file(remote_url, local_path)

    

    def load_model(self, model_name, remote_url_ckpt, remote_url_yaml, model_class):
        """Downloads the model and config if needed, loads them into memory, and moves the model to the specified device."""
        ckpt_path = os.path.join(self.model_folder, f'{model_name}.ckpt')
        yaml_path = os.path.join(self.model_folder, f'{model_name}.yaml')

        # Download model files if not present
        self.download_file_if_not_exists(remote_url_ckpt, ckpt_path)
        self.download_file_if_not_exists(remote_url_yaml, yaml_path)

        # Load configuration
        with open(yaml_path, 'r') as f:
            config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

        # Get valid arguments for the model constructor
        model_args = inspect.signature(model_class.__init__).parameters
        valid_config = {key: value for key, value in dict(config.model).items() if key in model_args}

        # If the model requires a 'config' argument, pass the full config object
        if 'config' in model_args:
            model = model_class(config=config)  # Pass full config if required
        else:
            model = model_class(**valid_config)  # Otherwise, pass filtered config

        model.load_state_dict(torch.load(ckpt_path))
        model = model.to(self.device)
        model.eval()

        return model, config



    def load_onnx_model(self, model_path, remote_url):
        """Downloads and initializes an ONNX model if not already present."""
        if not os.path.isfile(model_path):
            self.download_file_if_not_exists(remote_url, model_path)
        return ort.InferenceSession(model_path, providers=self.providers, provider_options=[{"device_id": 0}])

    def initialize_model_if_needed(self, model_name, options):
        """Loads a model only if it hasn't been initialized yet."""
        if model_name == "BSRoformer" and not hasattr(self, 'model_bsrofo'):
            print(f'Loading {model_name} into memory')
            bs_model_name = "model_bs_roformer_ep_368_sdr_12.9628" if options["BSRoformer_model"] == "ep_368_1296" else "model_bs_roformer_ep_317_sdr_12.9755"
            remote_url_ckpt = f'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/{bs_model_name}.ckpt'
            remote_url_yaml = f'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/{bs_model_name}.yaml'
            self.model_bsrofo, self.config_bsrofo = self.load_model(bs_model_name, remote_url_ckpt, remote_url_yaml, BSRoformer)

        elif model_name == "Kim_MelRoformer" and not hasattr(self, 'model_melrofo'):
            print(f'Loading {model_name} into memory')
            remote_url_ckpt = f'https://huggingface.co/KimberleyJSN/melbandroformer/resolve/main/MelBandRoformer.ckpt'
            remote_url_yaml = f'https://raw.githubusercontent.com/ZFTurbo/Music-Source-Separation-Training/main/configs/KimberleyJensen/config_vocals_mel_band_roformer_kj.yaml'
            self.model_melrofo, self.config_melrofo = self.load_model('Kim_MelRoformer', remote_url_ckpt, remote_url_yaml, MelBandRoformer)

        elif model_name == "SCNetXL" and not hasattr(self, 'model_scnet'):
            print(f'Loading {model_name} into memory')   
            remote_url_ckpt = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/model_scnet_ep_54_sdr_9.8051.ckpt'
            remote_url_yaml = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.13/config_musdb18_scnet_xl.yaml'
            self.model_scnet, self.config_scnet = self.load_model('model_scnet_ep_54_sdr_9.8051', remote_url_ckpt, remote_url_yaml, SCNet)

        elif model_name == "InstVoc" and not hasattr(self, 'model_mdxv3'):
            print(f'Loading {model_name} into memory')
            remote_url_ckpt = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt'
            remote_url_yaml = 'https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml'
            self.model_mdxv3, self.config_mdxv3 = self.load_model('MDX23C-8KFFT-InstVoc_HQ', remote_url_ckpt, remote_url_yaml, TFC_TDF_net)

        elif model_name == "VitLarge" and not hasattr(self, 'model_vl'):
            print(f'Loading {model_name} into memory')
            remote_url_ckpt = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt'
            remote_url_yaml = 'https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml'
            self.model_vl, self.config_vl = self.load_model('model_vocals_segm_models_sdr_9.77', remote_url_ckpt, remote_url_yaml, Segm_Models_Net)

        elif model_name == "VOCFT" and not hasattr(self, 'infer_session1'):
            print(f'Loading {model_name} into memory')
            model_path = os.path.join(self.model_folder, 'UVR-MDX-NET-Voc_FT.onnx')
            remote_url = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx'
            self.infer_session1 = self.load_onnx_model(model_path, remote_url)
            self.mdx_models1 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=2)

        elif model_name == "InstHQ5" and not hasattr(self, 'infer_session2'):
            print(f'Loading {model_name} into memory')
            model_path = os.path.join(self.model_folder, 'UVR-MDX-NET-Inst_HQ_5.onnx')
            remote_url = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_5.onnx'
            self.infer_session2 = self.load_onnx_model(model_path, remote_url)
            self.mdx_models2 = get_models('tdf_extra', load=False, device=self.device, vocals_model_type=3)

    @property
    def instruments(self):
        if not self.options.get('vocals_only', False):
            return ['bass', 'drums', 'other', 'vocals']
        else:
            return ['vocals']

    def separate_music_file(self, mixed_sound_array, sample_rate, current_file_number=0, total_files=0):
        """
        Implements the sound separation for a single sound file
        Inputs: Outputs from soundfile.read('mixture.wav')
            mixed_sound_array
            sample_rate

        Outputs:
            separated_music_arrays: Dictionary numpy array of each separated instrument
            output_sample_rates: Dictionary of sample rates separated sequence
        """



        separated_music_arrays = {}
        output_sample_rates = {}
        

        overlap_demucs = self.overlap_demucs
        overlap_MDX = self.overlap_MDX
        shifts = 0
        overlap = overlap_demucs


        vocals_model_names = [
            "BSRoformer",
            "SCNetXL"
            "Kim_MelRoformer",
            "InstVoc",
            "VitLarge",
            "VOCFT",
            "InstHQ5"
        ]

        vocals_model_outputs = []
        weights = []
        for model_name in vocals_model_names:
            if self.options.get(f"use_{model_name}", False):
                self.initialize_model_if_needed(model_name, self.options)

            if options[f"use_{model_name}"]:

                if model_name == "BSRoformer":
                    print(f'Processing vocals with {model_name} model...')
                    sources_bs = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_bsrofo, self.config_bsrofo, dim_t=1101, bigshifts=options["BigShifts"])
                    vocals_bs = match_array_shapes(sources_bs, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals_bs)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        self.model_bsrofo.cpu()
                        del self.model_bsrofo
                    del sources_bs
                    torch.cuda.empty_cache()
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "SCNetXL":
                    print(f'Processing vocals with {model_name} model...')
                    sources_scnet = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_musdb18_scnet_xl, self.config_musdb18_scnet_xl, dim_t=256, bigshifts=options["BigShifts"])
                    vocals_scnet = match_array_shapes(sources_scnet, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals_scnet)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        self.model_scnet.cpu()
                        del self.model_scnet  
                    del sources_scnet
                    torch.cuda.empty_cache()
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "Kim_MelRoformer":
                    print(f'Processing vocals with {model_name} model...')
                    sources_mel = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_melrofo, self.config_melrofo, dim_t=1101, bigshifts=options["BigShifts"])
                    vocals_mel = match_array_shapes(sources_mel, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals_mel)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        self.model_melrofo.cpu()
                        del self.model_melrofo
                    del sources_mel
                    torch.cuda.empty_cache()
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "InstVoc":
                    print(f'Processing vocals with {model_name} model...')
                    sources3 = demix_new_wrapper(mixed_sound_array.T, self.device, self.model_mdxv3, self.config_mdxv3, dim_t=2048, bigshifts=options["BigShifts"])
                    vocals3 = match_array_shapes(sources3, mixed_sound_array.T)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        self.model_mdxv3.cpu()
                        del self.model_mdxv3
                    del sources3
                    torch.cuda.empty_cache()
                    vocals_model_outputs.append(vocals3)
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "VitLarge":
                    print(f'Processing vocals with {model_name} model...')
                    vocals4, instrum4 = demix_full_vitlarge(mixed_sound_array.T, self.device, self.model_vl)#, self.config_vl, dim_t=512)
                    vocals4 = match_array_shapes(vocals4, mixed_sound_array.T)
                    vocals_model_outputs.append(vocals4)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        self.model_vl.cpu()
                        del self.model_vl
                    del vocals4
                    torch.cuda.empty_cache()
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "VOCFT":
                    print(f'Processing vocals with {model_name} model...')
                    overlap = overlap_MDX
                    vocals_mdxb1 = 0.5 * demix_wrapper(
                        mixed_sound_array.T,
                        self.device,
                        self.mdx_models1,
                        self.infer_session1,
                        overlap=overlap,
                        vc=1.021,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_mdxb1 += 0.5 * -demix_wrapper(
                        -mixed_sound_array.T,
                        self.device,
                        self.mdx_models1,
                        self.infer_session1,
                        overlap=overlap,
                        vc=1.021,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_model_outputs.append(vocals_mdxb1)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        del self.infer_session1, self.mdx_models1
                    del vocals_mdxb1
                    torch.cuda.empty_cache()
                    weights.append(options.get(f"weight_{model_name}"))

                elif model_name == "InstHQ5":
                    print(f'Processing vocals with {model_name} model...')
                    overlap = overlap_MDX
                    sources2 = 0.5 * demix_wrapper(
                        mixed_sound_array.T,
                        self.device,
                        self.mdx_models2,
                        self.infer_session2,
                        overlap=overlap,
                        vc=1.019,
                        bigshifts=options['BigShifts'] // 3
                    )
                    sources2 += 0.5 * -demix_wrapper(
                        -mixed_sound_array.T,
                        self.device,
                        self.mdx_models2,
                        self.infer_session2,
                        overlap=overlap,
                        vc=1.019,
                        bigshifts=options['BigShifts'] // 3
                    )
                    vocals_mdxb2 = mixed_sound_array.T - sources2
                    vocals_model_outputs.append(vocals_mdxb2)
                    if not options['large_gpu']:
                        print(f'Unloading {model_name} from memory')
                        del self.infer_session2, self.mdx_models2
                    del vocals_mdxb2, sources2
                    weights.append(options.get(f"weight_{model_name}"))
                    torch.cuda.empty_cache()

                else:
                    # No more model to process or unknown one
                    pass

        print('Processing vocals: DONE!')
        
        vocals_combined = np.zeros_like(vocals_model_outputs[0])

        for output, weight in zip(vocals_model_outputs, weights):
            vocals_combined += output * weight

        vocals_combined /= np.sum(weights)
        del vocals_model_outputs

        if options['use_VOCFT']:
            vocals_low = lr_filter(vocals_combined.T, 12000, 'lowpass') # * 1.01055  # remember to check if new final finetuned volume compensation is needed  !
            vocals_high = lr_filter(vocals3.T, 12000, 'highpass')
            vocals = vocals_low + vocals_high
        else:
            vocals = vocals_combined.T

        if options['filter_vocals'] is True:
                vocals = lr_filter(vocals, 50, 'highpass', order=8)
        
        # Generate instrumental
        instrum = mixed_sound_array - vocals
        
        if options['vocals_only'] is False:
            
            audio = np.expand_dims(instrum.T, axis=0)
            audio = torch.from_numpy(audio).type('torch.FloatTensor').to(self.device)
            all_outs = []
            print('Processing with htdemucs_ft...')
            i = 0
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_ft')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 1
            print('Processing with htdemucs...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
    
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 2
            print('Processing with htdemucs_6s...')
            overlap = overlap_demucs
            model = pretrained.get_model('htdemucs_6s')
            model.to(self.device)
            out = apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            # More stems need to add
            out[2] = out[2] + out[4] + out[5]
            out = out[:4]
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            i = 3
            print('Processing with htdemucs_mmi...')
            model = pretrained.get_model('hdemucs_mmi')
            model.to(self.device)
            out = 0.5 * apply_model(model, audio, shifts=shifts, overlap=overlap)[0].cpu().numpy() \
                  + 0.5 * -apply_model(model, -audio, shifts=shifts, overlap=overlap)[0].cpu().numpy()
       
            out[0] = self.weights_drums[i] * out[0]
            out[1] = self.weights_bass[i] * out[1]
            out[2] = self.weights_other[i] * out[2]
            out[3] = self.weights_vocals[i] * out[3]
            all_outs.append(out)
            model = model.cpu()
            del model
            gc.collect()
            out = np.array(all_outs).sum(axis=0)
            out[0] = out[0] / self.weights_drums.sum()
            out[1] = out[1] / self.weights_bass.sum()
            out[2] = out[2] / self.weights_other.sum()
            out[3] = out[3] / self.weights_vocals.sum()

            # other
            res = mixed_sound_array - vocals - out[0].T - out[1].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['other'] = (2 * res + out[2].T) / 3.0
            output_sample_rates['other'] = sample_rate
    
            # drums
            res = mixed_sound_array - vocals - out[1].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['drums'] = (res + 2 * out[0].T.copy()) / 3.0
            output_sample_rates['drums'] = sample_rate
    
            # bass
            res = mixed_sound_array - vocals - out[0].T - out[2].T
            res = np.clip(res, -1, 1)
            separated_music_arrays['bass'] = (res + 2 * out[1].T) / 3.0
            output_sample_rates['bass'] = sample_rate
    
            bass = separated_music_arrays['bass']
            drums = separated_music_arrays['drums']
            other = separated_music_arrays['other']
    
            separated_music_arrays['other'] = mixed_sound_array - vocals - bass - drums
            separated_music_arrays['drums'] = mixed_sound_array - vocals - bass - other
            separated_music_arrays['bass'] = mixed_sound_array - vocals - drums - other

        # vocals
        separated_music_arrays['vocals'] = vocals
        output_sample_rates['vocals'] = sample_rate

        # instrum
        separated_music_arrays['instrum'] = instrum

        return separated_music_arrays, output_sample_rates


def predict_with_model(options):

    output_format = options['output_format']
    output_extension = 'flac' if output_format == 'FLAC' else "wav"
    output_format = 'PCM_16' if output_format == 'FLAC' else options['output_format']
    
    for input_audio in options['input_audio']:
        if not os.path.isfile(input_audio):
            print('Error. No such file: {}. Please check path!'.format(input_audio))
            return
    output_folder = options['output_folder']
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)

    model = None
    model = EnsembleDemucsMDXMusicSeparationModel(options)

    for i, input_audio in enumerate(options['input_audio']):
        print('Go for: {}'.format(input_audio))
        audio, sr = librosa.load(input_audio, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        
        
        if options['input_gain'] != 0:
            audio = dBgain(audio, options['input_gain'])

        print("Input audio: {} Sample rate: {}".format(audio.shape, sr))
        result, sample_rates = model.separate_music_file(audio.T, sr, i, len(options['input_audio']))
        
        for instrum in model.instruments:
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.{}'.format(instrum, output_extension)
            if options["restore_gain"] is True: #restoring original gain
                result[instrum] = dBgain(result[instrum], -options['input_gain'])
            sf.write(output_folder + '/' + output_name, result[instrum], sample_rates[instrum], subtype=output_format)
            print('File created: {}'.format(output_folder + '/' + output_name))

        # instrumental part 1
        # inst = (audio.T - result['vocals'])
        inst = result['instrum']

        if options["restore_gain"] is True: #restoring original gain
            inst = dBgain(inst, -options['input_gain'])

        output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.{}'.format('instrum', output_extension)
        sf.write(output_folder + '/' + output_name, inst, sr, subtype=output_format)
        print('File created: {}'.format(output_folder + '/' + output_name))
        
        if options['vocals_only'] is False:
            # instrumental part 2
            inst2 = (result['bass'] + result['drums'] + result['other'])
            output_name = os.path.splitext(os.path.basename(input_audio))[0] + '_{}.{}'.format('instrum2', output_extension)
            sf.write(output_folder + '/' + output_name, inst2, sr, subtype=output_format)
            print('File created: {}'.format(output_folder + '/' + output_name))


# Linkwitz-Riley filter
def lr_filter(audio, cutoff, filter_type, order=6, sr=44100):
    audio = audio.T
    nyquist = 0.5 * sr
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
    sos = signal.tf2sos(b, a)
    filtered_audio = signal.sosfiltfilt(sos, audio)
    return filtered_audio.T

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
    if array_1.shape[1] > array_2.shape[1]:
        array_1 = array_1[:,:array_2.shape[1]] 
    elif array_1.shape[1] < array_2.shape[1]:
        padding = array_2.shape[1] - array_1.shape[1]
        array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
    return array_1

def dBgain(audio, volume_gain_dB):
    attenuation = 10 ** (volume_gain_dB / 20)
    gained_audio = audio * attenuation 
    return gained_audio



if __name__ == '__main__':
    start_time = time()
    print("started!\n")
    m = argparse.ArgumentParser()
    m.add_argument("--input_audio", "-i", nargs='+', type=str, help="Input audio location. You can provide multiple files at once", required=True)
    m.add_argument("--output_folder", "-r", type=str, help="Output audio folder", required=True)
    m.add_argument("--large_gpu", action='store_true', help="It will store all models on GPU for faster processing of multiple audio files. Requires 11 and more GB of free GPU memory.")
    m.add_argument("--single_onnx", action='store_true', help="Only use single ONNX model for vocals. Can be useful if you have not enough GPU memory.")
    m.add_argument("--cpu", action='store_true', help="Choose CPU instead of GPU for processing. Can be very slow.")
    m.add_argument("--overlap_demucs", type=float, help="Overlap of splited audio for light models. Closer to 1.0 - slower", required=False, default=0.1)
    m.add_argument("--overlap_VOCFT", type=float, help="Overlap of splited audio for heavy models. Closer to 1.0 - slower", required=False, default=0.1)
    m.add_argument("--overlap_InstHQ5", type=float, help="Overlap of splited audio for heavy models. Closer to 1.0 - slower", required=False, default=0.1)
    m.add_argument("--overlap_VitLarge", type=int, help="Overlap of splited audio for heavy models. Closer to 1.0 - slower", required=False, default=1)
    m.add_argument("--overlap_InstVoc", type=int, help="MDXv3 overlap", required=False, default=2)
    m.add_argument("--overlap_BSRoformer", type=int, help="BSRoformer overlap", required=False, default=2)
    m.add_argument("--overlap_SCNetXL", type=int, help="SCNetXL overlap", required=False, default=4)
    m.add_argument("--weight_InstVoc", type=float, help="Weight of MDXv3 model", required=False, default=3)
    m.add_argument("--weight_VOCFT", type=float, help="Weight of VOC-FT model", required=False, default=1)
    m.add_argument("--weight_InstHQ5", type=float, help="Weight of instHQ5 model", required=False, default=1)
    m.add_argument("--weight_VitLarge", type=float, help="Weight of VitLarge model", required=False, default=1)
    m.add_argument("--weight_BSRoformer", type=float, help="Weight of BS-Roformer model", required=False, default=8)
    m.add_argument("--weight_SCNetXL", type=float, help="Weight of SCNetXL model", required=False, default=10)
    m.add_argument("--weight_Kim_MelRoformer", type=float, help="Weight of Kim_MelRoformer model", required=False, default=10)
    m.add_argument("--BigShifts", type=int, help="Managing MDX 'BigShifts' trick value.", required=False, default=3)
    m.add_argument("--vocals_only",  action='store_true', help="Vocals + instrumental only")
    m.add_argument("--use_BSRoformer", action='store_true', help="use BSRoformer in vocal ensemble")
    m.add_argument("--use_Kim_MelRoformer", action='store_true', help="use Kim MelBand Roformer in vocal ensemble")
    
    m.add_argument("--BSRoformer_model", type=str, help="Which checkpoint to use", required=False, default="ep_317_1297")
    m.add_argument("--use_SCNetXL", action='store_true', help="use SCNetXL in vocal ensemble")
    m.add_argument("--use_InstVoc", action='store_true', help="use instVoc in vocal ensemble")
    m.add_argument("--use_VitLarge", action='store_true', help="use VitLarge in vocal ensemble")
    m.add_argument("--use_InstHQ5", action='store_true', help="use InstHQ5 in vocal ensemble")
    m.add_argument("--use_VOCFT", action='store_true', help="use VOCFT in vocal ensemble")
    m.add_argument("--output_format", type=str, help="Output audio folder", default="PCM_16")
    m.add_argument("--input_gain", type=int, help="input volume gain", required=False, default=0)
    m.add_argument("--restore_gain", action='store_true', help="restore original gain after separation")
    m.add_argument("--filter_vocals", action='store_true', help="Remove audio below 50hz in vocals stem")
    options = m.parse_args().__dict__
    print("Options: ")
    print(f'large_gpu: {options["large_gpu"]}\n')
    print(f'Input Gain: {options["input_gain"]}dB')
    print(f'Restore Gain: {options["restore_gain"]}')
    print(f'BigShifts: {options["BigShifts"]}\n')

    print(f'BSRoformer_model: {options["BSRoformer_model"]}')
    print(f'weight_BSRoformer: {options["weight_BSRoformer"]}')
    print(f'weight_InstVoc: {options["weight_InstVoc"]}\n')

    print(f'use_VitLarge: {options["use_VitLarge"]}')
    if options["use_VitLarge"] is True:    
       print(f'weight_VitLarge: {options["weight_VitLarge"]}\n')
    
    print(f'use_VOCFT: {options["use_VOCFT"]}')
    if options["use_VOCFT"] is True:
        print(f'overlap_VOCFT: {options["overlap_VOCFT"]}')
        print(f'weight_VOCFT: {options["weight_VOCFT"]}\n')
        
    print(f'use_InstHQ5: {options["use_InstHQ5"]}')
    if options["use_InstHQ5"] is True:
        print(f'overlap_InstHQ5: {options["overlap_InstHQ5"]}')
        print(f'weight_InstHQ5: {options["weight_InstHQ5"]}\n')

    print(f'vocals_only: {options["vocals_only"]}')
    
    if options["vocals_only"] is False:
        print(f'overlap_demucs: {options["overlap_demucs"]}\n')

    print(f'output_format: {options["output_format"]}\n')
    predict_with_model(options)
    print('Time: {:.0f} sec'.format(time() - start_time))

