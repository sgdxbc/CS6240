from pathlib import Path
import tensorflow as tf

env = "development"
current_path = Path()
if (current_path / "config_production.py").exists():
    env = "production"

###
# common settings
###
model_path = current_path / "classifier_model_42"
data_dir = current_path / "data" / "mini_speech_commands"
# list of (<what command we want to be misunderstood>, <what it should be understood>)
target_map = [("left", "right")]
# how many times we care loudness more than accuracy
alpha = 50.0
SAMPLE_RATE = 16 * 1000
sample_interval = 10 * (SAMPLE_RATE // 1000)

## graphic card memory requirement ~ len(target_map) * train_count * batch_size
# training/validation audio from each class of target_map
train_count = 10
val_count = 10
# how much we want the task to be parallelized
batch_size = 1
# how much progress we want in one epoch
batch_per_epoch = 80 // batch_size

opt = tf.keras.optimizers.Adam()
stop_when_stable_in = 20  # epoches
stable_std = 1e-5

###
# flipping attack
###
flipping_settings = dict(
    perturbation_length=600 * (SAMPLE_RATE // 1000),
    output_path=current_path / "adv.wav",
)

###
# music attack
###
music_settings = dict(
    perturbation_chunk_length=50 * (SAMPLE_RATE // 1000),
    chunk_count=3,
    output_path=current_path / "pert.wav",
    music_path=current_path / "underlay.wav",
)

if env == "production":
    from config_production import *