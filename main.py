from landmarks_extractor import LandmarkDetector
from helpers import *
from animator import Animator
from bg_remover import *
import os
import json
import pandas as pd


# np.seterr(all="ignore")

def PostDetection_Pipeline(image_input, audio_input, audio_directory, outputFolder, inputFolder):
    a = Animator(inputImage=image_input, inputAudio=audio_input,
                 outputFolder=outputFolder, audio_dir=audio_directory)

    # Create embeddings for audios
    ains, au_emb, au_data = a.GenerateAudioInput()

    print("[+] Audio processed")

    # Load  and clean facial Landmark data
    facial_landmark_data = a.GetFacialLandmarkData(au_data)
    print("fl_data")
    print(facial_landmark_data)

    print("[+] Facial Landmark Data Loaded")

    # Generate new landmarks for each audio sample
    a.AudioToLandmark(au_emb)
    print("[+] New Landmarks Generated")

    # Use Facewarp to create final frames / Stitch the frames
    a.DenormalizeOutputToOriginalImage()
    print("[+] Output Denormalized")


if __name__ == "__main__":



    # Landmarker.edit()

    # initCleanLog()

    # images = ["ape1", "ape2", "ape3", "ape4", "ape5"]
    image_input = "ape2"
    audio_input = "obama"
    audio_directory = "audio"
    outputFolder = "ape_src"
    inputFolder = "ape_src"

    PostDetection_Pipeline(image_input, audio_input, audio_directory, outputFolder, inputFolder)


    # Landmarker = LandmarkDetector(inputImage=f"{image_input}.jpg")
    # Landmarker.detect()
    # Landmarker.edit()

    

    


    