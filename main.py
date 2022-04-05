from landmarks_extractor import LandmarkDetector
from helpers import *
from animator import Animator
import os
import json
import pandas as pd

np.seterr(all="ignore")


if __name__ == "__main__":

    # AnimateMouth("Bored", 1, "./IO/sample.wav")

    # Landmarker = LandmarkDetector(
    #     inputImage=f"{image_input}.jpg")
    # detected_landmarks = Landmarker.detect()

    # Landmarker.edit()

    initCleanLog()

    image_input = "ape5"
    audio_input = "Welcome"
    audio_directory = "audio"
    outputFolder = "ape_src"

    # cp_landmarks_for_mouth("ape2", "ape5")

    a = Animator(inputImage=image_input, inputAudio=audio_input,
                 outputFolder=outputFolder, audio_dir=audio_directory)

    # Create embeddings for audios
    ains, au_emb, au_data = a.GenerateAudioInput()

    print("[+] Audio processed")

    # Load  and clean facial Landmark data
    facial_landmark_data = a.GetFacialLandmarkData(au_data)

    print("[+] Facial Landmark Data Loaded")

    # Generate new landmarks for each audio sample
    a.AudioToLandmark(au_emb)
    print("[+] New Landmarks Generated")

    # Use Facewarp to create final frames / Stitch the frames
    a.DenormalizeOutputToOriginalImage()
    print("[+] Output Denormalized")

