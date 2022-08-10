from helpers import *
from animator import Animator
from bg_remover import *
import os
import json
import pandas as pd

if __name__ == "__main__":

    char_name = "doll_pose"
    image_input_dir = "./input/character_data/0002_doll_pose"
    audio_name = "simp"

    MakeItTalk_Inference(char_name, audio_name, image_input_dir)
