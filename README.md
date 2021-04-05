# Project setup

Follow [pip setup](https://www.tensorflow.org/install/pip). Notice: CUDA version must be 11.0, not the latest.

To train audio command classifier:
* Open `train.py`.
* Edit `seed` and `save_path` near the top of the code.
* Run `python train.py`.

To generate adversarial sample against trained classifier:
* Open `generate.py`.
* Edit parameters at the top as needed and run.

The predict and mixing script is been refactoring.

Work in progress.
