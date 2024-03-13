import subprocess
from random import random

import IPython.display as ipd

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

import os
import IPython
from IPython.display import Audio as play
from pathlib import Path
import time


class TrumpSpeak:

    voices = [
        "Brian",
        "Emma",
        "Russell",
        "Joey",
        "Matthew",
        "Joanna",
        "Kimberly",
        "Amy",
        "Geraint",
        "Nicole",
        "Justin",
        "Ivy",
        "Kendra",
        "Salli",
        "Raveena",
    ]

    def __init__(self):
        self.url = "https://streamlabs.com/polly/speak"
        self.max_chars = 800
        self.voices = []

    def run(self, text, filepath, random_voice: bool = False):

        print("\n\n")
        path = os.path.abspath("model_outputs/ljspeech_tts.forward/")

        model_iteration = 80  # @param {type:"slider", min:40, max:100, step:20}
        talking_speed = 0.7  # @param {type:"slider", min:0.5, max:1, step:0.01}
        modelWeights = str(model_iteration) + "K.pyt"

        # Use subprocess.run with a list of arguments
        subprocess.run([
            "python",
            "gen_forward.py",
            "--alpha", str(talking_speed),
            "--input_text", text,
            "--outpath", filepath,
            "--hp_file", "pretrained/pretrained_hparams.py",
            "--tts_weights", f"checkpoints/ljspeech_tts.forward/{modelWeights}",
            "wavernn",
            "--voc_weights", "pretrained/wave_800K.pyt",
            "--batched",
            "--target", "4096",
            "--overlap", "32"
        ])

    def randomvoice(self):
        return random.choice(self.voices)



