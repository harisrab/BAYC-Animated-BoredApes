from landmarks_extractor import LandmarkDetector
from helpers import *
from cartoonize import *
import os
import json
import pandas as pd

np.seterr(all="ignore")


def AnimateMouth(mouth, amount, audio):
    """Takes in mouth type, the number of random faces, the audio used"""
    apes = pd.read_csv("./ape_database/apes.csv")
    filtered_apes = apes[(apes["Mouth"] == mouth)].sample(n=amount)

    # Current ape
    # ape = filtered_apes["image"].iloc[0]
    ape = "ape2"

    # Augment the ape
    img = augment_ape(ape)

    # Run landmark detector
    Landmarker = LandmarkDetector(
        inputImage=f"{ape}.jpg", src_dir="ape_database/ipfs", out_dir="ape_src")

    # # Run landmarks detection algorithm
    Landmarker.detect()

    # # Open up interface for editing landmarks
    Landmarker.edit()


def updateFrontend():
    db = pd.read_csv("./ape_database/apes.csv")

    apes_list = db["image"].to_list()

    with open('./frontend/public/apes_availability.csv', "w") as apes:
        apes.write(f"ape_code,availablity\n")

        for ape_name in apes_list:
            if(os.path.exists(os.path.join("./ape_database/ipfs", ape_name + '.pts'))):
                apes.write(f"{ape_name},yes\n")


if __name__ == "__main__":

    # updateFrontend()

    # AnimateMouth("Bored", 1, "./IO/sample.wav")

    image_input = "ape2"
    #image_input = "Qma1aZPn7iS1vxkfip6kjGjbA5EUPDaunsApJJ8mUt8pyT"
    audio_input = "M6_04_16k"

    # Landmarker = LandmarkDetector(
    #     inputImage=f"{image_input}.jpg")
    # detected_landmarks = Landmarker.detect()

    # Landmarker.edit()

    cartoonize = Cartoonizer(
        image=image_input, bg=f"{image_input}_bg.jpg", audioName=audio_input)

    # print(detected_landmarks)
    # os.system("")
    # os.system("ffmpeg -i ape1.mp4 -i ape2.mp4 -i ape3.mp4 -i ape4.mp4 -i ape5.mp4 -filter_complex hstack=inputs=5 horizontal-stacked-output.mp4")

    # Stack generated videos <ape1.mp4>, etc
    # os.system("ffmpeg -i ape1.mp4 -i ape2.mp4 -i ape3.mp4 -i ape4.mp4 -i ape5.mp4 -filter_complex hstack=inputs=5 horizontal-stacked-output.mp4")

    # # Loop the stack
    # os.system("ffmpeg -stream_loop 10 -i horizontal-stacked-output.mp4 -c copy repeated_output.mp4")

    # # Get length of stacked video
    # os.system("ffprobe -v error -select_streams v:0 -show_entries stream=duration -of default=noprint_wrappers=1:nokey=1 repeated_output.mp4")

    # # Stack .jpgs
    # os.system("ffmpeg -i ape1.jpg -i ape2.jpg -i ape3.jpg -i ape4.jpg -i ape5.jpg -filter_complex hstack=inoutput.jpg")

    # # Loop .jpgs to original videos length
    # os.system("ffmpeg -loop 1 -i image.png -c:v libx264 -t 11.952625 -pix_fmt yuv420p pic_vid.mp4")

    # # Stack the static and video
    # os.system("ffmpeg -i pic_vid.mp4 -i horizontal-stacked-output.mp4 -filter_complex vstack=inputs=2 complete-vertical-stack-output.mp4")
