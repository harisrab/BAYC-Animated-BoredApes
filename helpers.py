import os
import shutil
import json
import pandas as pd
import cv2
import numpy as np


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


