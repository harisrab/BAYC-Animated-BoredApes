from typing import Dict, Optional, List
from PIL import Image
  
  


def LoadImage(ape_id: str):

    # open method used to open different extension image file
    ape_img = Image.open(r"../ape_database/ape_src/" + ape_id) 

    pass


def LoadLandmarks() -> List:
    pass