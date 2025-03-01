from train_codes.utils.audio import load_wav, melspectrogram
import numpy as np
if __name__ == "__main__":
    wav_path = './data/audio/yongen.wav'
    try:
        wav = load_wav(wav_path, 16000).T
        print("successfully load wav")
        ori_mel = melspectrogram(wav).T
        print(f"ori_mel: {ori_mel.shape}")
        concatenated_mel = np.concatenate([ori_mel, ori_mel], axis=0)
        print(f"concatenated_mel: {concatenated_mel.shape}")
    except Exception as e:
        print(e)
    