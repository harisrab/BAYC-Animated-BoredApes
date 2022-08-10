import os
import shutil
import json
import pandas as pd
import cv2
import numpy as np
from animator import Animator
from bg_remover import *

def initCleanLog():
    # print(os.system("pwd"))
    os.system("touch log/audio_processing_log.txt")
    os.system("touch log/ffmpeg_stitch_log.txt")


def updateFrontend():
    db = pd.read_csv("./ape_database/apes.csv")

    apes_list = db["image"].to_list()

    with open('./frontend/public/apes_availability.csv', "w") as apes:
        apes.write(f"ape_code,availablity\n")

        for ape_name in apes_list:
            if(os.path.exists(os.path.join("./ape_database/ipfs", ape_name + '.pts'))):
                apes.write(f"{ape_name},yes\n")



def cp_landmarks_for_mouth(ape1, ape2):
    shutil.copy2("./ape_src/{}.pts".format(ape1), "./ape_src/{}.pts".format(ape2))
    shutil.copy2("./ape_src/{}_close_mouth.txt".format(ape1), "./ape_src/{}_close_mouth.txt".format(ape2))
    shutil.copy2("./ape_src/{}_delauney_tri.txt".format(ape1), "./ape_src/{}_delauney_tri.txt".format(ape2))
    shutil.copy2("./ape_src/{}_face_close_mouth.txt".format(ape1), "./ape_src/{}_face_close_mouth.txt".format(ape2))
    shutil.copy2('./ape_src/{}_face_open_mouth.txt'.format(ape1), "./ape_src/{}_face_open_mouth.txt".format(ape2))
    shutil.copy2('./ape_src/{}_face_open_mouth_norm.txt'.format(ape1), "./ape_src/{}_face_open_mouth_norm.txt".format(ape2))
    shutil.copy2('./ape_src/{}_face_open_mouth.txt'.format(ape1), "./ape_src/{}_open_mouth.txt".format(ape2))
    shutil.copy2('./ape_src/{}_open_mouth_norm.txt'.format(ape1), "./ape_src/{}_open_mouth_norm.txt".format(ape2))
    shutil.copy2('./ape_src/{}_scale_shift.txt'.format(ape1), "./ape_src/{}_scale_shift.txt".format(ape2))


def json_to_df(filepath):
    """ Takes in JSON and outputs an excel file for easy APE query """

    with open(filepath) as json_file:
        apes = json.load(json_file)["apes"]
        newApes = []

        print("[+] Parsing JSON")
        for i, ape in enumerate(apes):
            ape_tmp = {
                "id": ape["_id"],
                "transactionHash": ape["transactionHash"],
                "blockNumber": ape["blockNumber"],
                "image": ape["metadata"]["image"][7:]
            }

            for i in range(len(ape["metadata"]["attributes"])):
                ape_tmp = {
                    **ape_tmp, ape["metadata"]["attributes"][i]["trait_type"]: ape["metadata"]["attributes"][i]["value"],
                }

            newApes.append(ape_tmp)

        print("[+] JSON Parse Complete")

        apes_df = pd.DataFrame(newApes)

        apes_df.to_csv('./ape_database/apes.csv', index=False)

        print("[+] Saved CSV to ./ape_database/apes.csv")


def augment_ape(img_code):
    """
    1. Pick the image
    2. resize the image (256 x 256)
    3. Pick the background color
    4. Remove the background
    5. Return cv ape face and bg
    """
    # Read the image
    img = cv2.imread(f'./ape_database/ipfs/{img_code}.jpg', 1)
  
    # Resize the image
    resized_img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)

    return resized_img

def MakeItTalk_Inference(char_name, audio_name, image_input_dir):
    
    # Initialize the data required for inference
    a = Animator(char_name, audio_name, image_input_dir)
    
    print("[+] Animator Pipeline Initialized")

    # Process the audio data
    ains, au_emb, au_data = a.GenerateAudioInput()

    print("[+] Audio Processed")

    # Load Facial Landmarks Data
    facial_landmark_data = a.GetFacialLandmarkData(au_data)

    print("[+] Facial Landmark Data Loaded")

    #Generate new landmarks for each audio sample
    a.AudioToLandmark(au_emb)

    print("[+] New Landmarks Generated")

    # Use Facewarp to create final frames / stitch them up to output.
    a.DenormalizeOutputToOriginalImage()

    print("[+] Output Denormalized")

