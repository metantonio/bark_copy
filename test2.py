from transformers import AutoProcessor, BarkModel
import scipy
import os
os.environ["SUNO_OFFLOAD_CPU"] = "true"
os.environ["SUNO_USE_SMALL_MODELS"] = "true"

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello Andrea", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()


sample_rate = model.generation_config.sample_rate
scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=audio_array)
