import os
from PIL import Image

def GenerateTransparentApe(input_img, output_path):
    os.chdir("./bgRemover")
    os.system(
        f"./remove_bg {input_img} {output_path}"
    )
    os.chdir("../")


def CreateBG():
    colors = ["AQUAMARINE", "ARMY GREEN", "BLUE", "GRAY", "NEW PUNK BLUE", "PURPLE", "ORANGE", "YELLOW"]
    hex_codes = ["#17e6b7", "#717234", "#a2e5f4", "#cccdce", "#3a677d", "#ef972c", "#ef972c", "#e4e4a8"]

    for color, hex_code in zip(colors, hex_codes):

        tmp = Image.new(mode="RGB", size=(256,256), color=hex_code)
        tmp.save(f"./ape_database/bg/{color}.jpg")
 
 
 
 
 
 
 