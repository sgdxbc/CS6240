from pathlib import Path
import tensorflow as tf
from utils import *

data_dir = Path() / "data" / "mini_speech_commands"
# audio_path = data_dir / "right" / "0ab3b47d_nohash_0.wav"
audio_path = Path() / "right2left(mix).wav"
model_path = Path() / "classifier_model_42"

spectrogram = get_spectrogram(decode_audio(tf.io.read_file(str(audio_path))))
spectrogram = tf.expand_dims(spectrogram, -1)
spectrogram = tf.expand_dims(spectrogram, 0)
model = tf.keras.models.load_model(str(model_path))
model.summary()
print(get_commands(data_dir))
print(model.predict(spectrogram))