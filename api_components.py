import os
import uuid
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from io import BytesIO



parent_path = Path(__file__).parent.absolute()
common_dir = 'common'
common_path = os.path.join(parent_path, common_dir)

def read_imagefile(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image

def crop_n_save(coordinates,input_image):
    if not os.path.isdir(common_path):
        os.mkdir(common_path)

    temp_path = os.path.join(common_path,str(uuid.uuid4())[:8])
    os.mkdir(temp_path)
    file_path_list = []
    if(not coordinates['detections'] == -1):
        coordinates_list = coordinates['detections']['labels']
        i=1
        for code in coordinates_list:
            img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)
            crop_img = img[code['Y']:code['Y']+code['Height'], code['X']:code['X']+code['Width']]
            file_name = 'image_'+str(i)+'.jpg'
            save_path = os.path.join(temp_path,file_name)
            cv2.imwrite(save_path,crop_img)
            file_path_list.append(save_path)
            i=i+1
    
    return file_path_list

