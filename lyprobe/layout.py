"""
Lychee-FD input layout while the model is listening, as used in training.

Read from HITsz-TMG/Lychee-FD @7bb1068:
  Training_package/Code/DataLoaders/LycheeFDDataset.py   (listening chunks, audio grid)
  Training_package/Code/Models/LycheeFD.py               (how the channels are summed)
  HF HIT-TMG/Lychee-FD config.json                       (token ids, chunk size)

Sequence = [system prefix][suffix]. Every suffix slot is 40 ms and carries four
channels whose embeddings are summed:

  text     text_pad                                     x10 per chunk
  speech   stoken_pad                                   x10 per chunk
  control  [sleep x8][detect][keep_listening]           per chunk
  audio    [<audio_patch>, audio_pad] x5                per chunk

One chunk = 10 slots = 0.4 s. User audio is cut into 0.4 s windows, each window gets
its own log-mel and goes through the encoder separately, giving 5 features at 12.5 Hz;
`align_audio_input` (always True in training) places them on the even slots.

Hard deps: numpy
"""

import numpy as np

SR = 16000
SLOT_S = 0.04
CHUNK = 10                       # control_token_chunk_size
CHUNK_S = CHUNK * SLOT_S         # 0.4 s
FEATS_PER_CHUNK = CHUNK // 2     # audio features per window (align_audio_input)

SYSTEM_MESSAGE_PREFIX = "<|BOT|>system\nYou are a helpful assistant.<|EOT|>"

# token ids from the released config.json / added_tokens.json
TOK = {
    "audio_patch": 151690,   # placeholder replaced by encoder features (hard-coded in LycheeFD.forward)
    "audio_pad": 158360,
    "text_pad": 158358,
    "stoken_pad": 158359,
    "sleep": 158357,
    "detect": 158356,
    "keep_listening": 158354,
}


def mel_frames(n_samples, padding=479, hop=160):
    """Frames of Lychee's log_mel_spectrogram: torch.stft(center=True) then drop the last."""
    return (n_samples + padding) // hop


def compute_token_num(max_feature_len):
    """Copied from LycheeFDDataset.compute_token_num: encoder (x/4) then adaptor (stride 2)."""
    max_feature_len = max_feature_len - 2
    encoder_output_dim = (max_feature_len + 1) // 2 // 2
    padding, kernel_size, stride = 1, 3, 2
    return (encoder_output_dim + 2 * padding - kernel_size) // stride + 1


def listening_suffix(n_chunks, tok=TOK):
    """Return the four suffix channels (lists of ids) for n_chunks of listening."""
    text = [tok["text_pad"]] * (CHUNK * n_chunks)
    stoken = [tok["stoken_pad"]] * (CHUNK * n_chunks)
    control = ([tok["sleep"]] * (CHUNK - 2) + [tok["detect"], tok["keep_listening"]]) * n_chunks
    audio = [tok["audio_patch"], tok["audio_pad"]] * (FEATS_PER_CHUNK * n_chunks)
    return text, stoken, control, audio


def prepare_waveform(x, sr, lead_chunks, trim_fn):
    """Trim, add lead silence, pad to a chunk boundary.

    Returns (wave, lead_s, speech_s, n_chunks). Speech starts on a chunk boundary, so
    every slot can be labeled speech / silence from known offsets.
    """
    speech, _, _ = trim_fn(x, sr)
    chunk_n = int(round(CHUNK_S * sr))
    lead = np.zeros(lead_chunks * chunk_n, np.float32)
    body = np.concatenate([lead, speech])
    n_chunks = int(np.ceil(len(body) / chunk_n))
    wave = np.zeros(n_chunks * chunk_n, np.float32)
    wave[:len(body)] = body
    return wave, len(lead) / sr, len(speech) / sr, n_chunks


def speech_slot_mask(n_slots, lead_s, speech_s, slot_s=SLOT_S, min_overlap=0.5):
    """True for slots whose 40 ms window overlaps the speech span by >= min_overlap."""
    start = np.arange(n_slots) * slot_s
    end = start + slot_s
    overlap = np.clip(np.minimum(end, lead_s + speech_s) - np.maximum(start, lead_s), 0, None)
    return overlap >= min_overlap * slot_s


def window_frame_mask(n_windows, frames_per_window, frame_s, lead_s, speech_s,
                      window_s=CHUNK_S, min_overlap=0.5):
    """Encoder frames inside independently encoded windows -> [n_windows, frames] mask.

    Frame j of window i covers [i * window_s + j * frame_s, + frame_s). True where that
    span overlaps the speech by >= min_overlap of a frame.
    """
    start = (np.arange(n_windows)[:, None] * window_s + np.arange(frames_per_window)[None, :] * frame_s)
    end = start + frame_s
    overlap = np.clip(np.minimum(end, lead_s + speech_s) - np.maximum(start, lead_s), 0, None)
    return overlap >= min_overlap * frame_s - 1e-9
