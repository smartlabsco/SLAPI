from fastapi import FastAPI, File, UploadFile, Header, HTTPException, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import logging
import logging.config
import json
import os
import io
import sys
from PIL import Image
from enum import Enum
import cv2
import numpy as np
import base64
import shutil
from datetime import datetime
sys.path.append('./sl_faceswap')
from sl_image4_1_f import FaceswapProcessor
from sl_image4 import FaceswapParser



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


processor = FaceswapProcessor(base_path='./sl_faceswap')
log_config = json.load(open('log.json'))
logging.config.dictConfig(log_config)
logger = logging.getLogger(__name__)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
auth_file = "auth.ini"
with open(auth_file) as f:
    auth_list = f.read().splitlines()

def auth(api_key: str = Header(None)):
    if api_key not in auth_list:
        raise HTTPException(status_code=403, detail="Auth error")
    
@app.get("/")
def welcome():
    """
    A simple endpoint that returns a welcome message.

    Returns:
        str: A welcome message.
    """
    return 'Welcome smartlabs openAPI'

class TypeEnum(str, Enum):
    asian_m = "filter1"
    asian_g = "filter2"
    western_m = "filter3"
    western_g = "filter4" 
    black_m = "filter5"
    black_g = "filter6"
@app.post("/parsingImgImg")
async def parsingImgImg(file: UploadFile = File(...), type: TypeEnum = Form(...),api_key: str = Header(None)):


    try:
        auth(api_key)
        UPLOAD_DIR = "./sl_faceswap/data/parsingimg_1/"
        filename = "UPLOAD_IMG.png"
        content =await file.read()
        with open(os.path.join(UPLOAD_DIR + "target", filename), "wb") as fp:
            fp.write(content)

        os.chdir('sl_faceswap')

        with open("type.txt", "w") as type_file:
          type_file.write(type)

        os.system('python sl_image.1.py')
        os.chdir('../')
        source_path = "./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png"
        destination_dir = '/home/smartlabs/apitest/newweb_test/static'
        destination_path = os.path.join(destination_dir, 'PARSE_OUTPUT.png')
        if os.path.exists(destination_path):
            backup_path = destination_path + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
            shutil.copyfile(destination_path, backup_path)
            print(f"Backup of the existing file created at: {backup_path}")
            
        shutil.copy(source_path, destination_path)
        print(f"File moved and overwritten if already existed at: {destination_path}")
        

        with open("./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
        img_data = text
        im = Image.open("./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png")
        return JSONResponse(content={"msg": "success", "img_size": [im.size[0], im.size[1]], "img_data": img_data}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingImgImg2")
async def parsingImgImg2(source: UploadFile = File(...), target: UploadFile = File(...), api_key: str = Header(None)):
    


    try:
        auth(api_key)

        UPLOAD_DIR = "./sl_faceswap/data/parsingimg_2/"
        filename1 = "UPLOAD_IMG_source.png"
        content1 = await source.read()
        with open(os.path.join(UPLOAD_DIR + "source", filename1), "wb") as fp:
            fp.write(content1)
        filename2 = "UPLOAD_IMG_target.png"
        content2 = await target.read()
        with open(os.path.join(UPLOAD_DIR + "target", filename2), "wb") as fp:
            fp.write(content2)

        os.chdir('sl_faceswap')
        os.system('python sl_image_2.py')
        os.chdir('../')

        with open("./sl_faceswap/data/parsingimg_2/result/PARSE_OUTPUT.png", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
        img_data = text
        im = Image.open("./sl_faceswap/data/parsingimg_2/result/PARSE_OUTPUT.png")
        return JSONResponse(content={"msg": "success", "img_size": [im.size[0], im.size[1]], "img_data": img_data}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingImgImg3")
async def parsingImgImg(file: UploadFile = File(...), type: TypeEnum = Form(...),api_key: str = Header(None)):



    try:
        auth(api_key)
        UPLOAD_DIR = "./sl_faceswap/data/parsingimg_1/"
        filename = "UPLOAD_IMG.png"
        content =await file.read()
        with open(os.path.join(UPLOAD_DIR + "target", filename), "wb") as fp:
            fp.write(content)

        os.chdir('sl_faceswap')

        with open("type.txt", "w") as type_file:
          type_file.write(type)

        os.system('python sl_image.3.py')
        os.chdir('../')
        source_path = "./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png"
        destination_dir = '/home/smartlabs/apitest/newweb_test/static'
        destination_path = os.path.join(destination_dir, 'PARSE_OUTPUT.png')
        if os.path.exists(destination_path):
            backup_path = destination_path + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
            shutil.copyfile(destination_path, backup_path)
            print(f"Backup of the existing file created at: {backup_path}")
            
        shutil.copy(source_path, destination_path)
        print(f"File moved and overwritten if already existed at: {destination_path}")


        suc_value=''

        try:
            with open("sl_faceswap/suc.json", "r") as file:
                data = json.load(file)
                suc_value = data.get("SUC", "Error: SUC key not found")
        except FileNotFoundError:
            suc_value = "Error: File not found"



        with open("./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
        img_data = text
        im = Image.open("./sl_faceswap/data/parsingimg_1/result/PARSE_OUTPUT.png")
        return JSONResponse(content={"msg": "success", "img_size": [im.size[0], im.size[1]], "img_data": img_data, "SUC": suc_value}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingImgImg4")
async def parsingImgImg4(file: UploadFile = File(...), api_key: str = Header(None)):


    try:
        auth(api_key)
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        os.chdir('sl_faceswap')
        parser = FaceswapParser()
        result_img, bbox_list = parser.parsing(image, None)
        os.chdir('../')

        if result_img is None:
            raise HTTPException(status_code=500, detail="Failed to process image")
        _, buffer = cv2.imencode('.png', result_img)
        img_str = base64.b64encode(buffer).decode('ascii')
        height, width = result_img.shape[:2]
        response_data = {
            "msg": "success",
            "img_size": [width, height],
            "img_data": img_str,
            "bboxes": bbox_list
        }

        return JSONResponse(content=response_data, status_code=200)
        
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingImgImg4_1")
async def parsingImgImg4_1(
    file: UploadFile = File(...),
    filter: str = Form(...),
    api_key: str = Header(None)
):
    try:
        auth(api_key)
        content = await file.read()
        nparr = np.frombuffer(content, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        try:
            filter_data = json.loads(filter)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail=f"Invalid JSON format in filter: {str(e)}")
        result_img = processor.process_image(image, filter_data)
        _, buffer = cv2.imencode('.png', result_img)
        img_str = base64.b64encode(buffer).decode()

        return JSONResponse(content={
            "msg": "success",
            "img_size": [result_img.shape[1], result_img.shape[0]],
            "img_data": img_str
        }, status_code=200)
    except Exception as e:
        logger.error(f"Error in parsing_img_img4_1: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
@app.post("/parsingLicense1")
async def parsingLicense1(file: UploadFile = File(...), api_key: str = Header(None)):
    
    try:
        auth(api_key)

        UPLOAD_DIR = "./bunho_swap/data/"
        filename = "UPLOAD_IMG.png"
        content =await file.read()
        with open(os.path.join(UPLOAD_DIR + "target", filename), "wb") as fp:
            fp.write(content)

        os.chdir('bunho_swap')
        os.system('python detect_lp_detect.py')
        os.chdir('../')
        

        with open("./bunho_swap/data/target/UPLOAD_IMG.png", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
        img_data = text
        im = Image.open("./bunho_swap/data/target/UPLOAD_IMG.png")
        try:
            with open('./bunho_swap/data/result/bboxes.json', 'r') as json_file:
                bbox_data = json.load(json_file)
        except FileNotFoundError:
            bbox_data = []

        
        response_data = {
            "msg": "success",
            "img_size": [im.size[0], im.size[1]],
            "img_data": img_data,
            "bboxes": bbox_data
        }

        return JSONResponse(content=response_data, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingLicense1_1")
async def parsingLicense1_1(file: UploadFile = File(...), filter: str = Form(...), api_key: str = Header(None)):
    print("here")



    try:
        auth(api_key)
        UPLOAD_DIR = "./bunho_swap/data/"
        filename = "UPLOAD_IMG.png"
        content =await file.read()
        with open(os.path.join(UPLOAD_DIR + "target", filename), "wb") as fp:
            fp.write(content)

        os.chdir('bunho_swap')
        try:
            filter_data = json.loads(filter)
            with open('filter.json', 'w', encoding='utf-8') as f:
                json.dump(filter_data, f, ensure_ascii=False, indent=4)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON format in filter: {e}"}

        os.system('python detect_lp_lobs.py')
        os.chdir('../')
        

        with open("./bunho_swap/data/result/PARSE_OUTPUT.png", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
        img_data = text
        im = Image.open("./bunho_swap/data/result/PARSE_OUTPUT.png")
        return JSONResponse(content={"msg": "success", "img_size": [im.size[0], im.size[1]], "img_data": img_data}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)
@app.post("/parsingImgVid")
async def parsingImgVid(file: UploadFile = File(...), type: TypeEnum = Form(...), api_key: str = Header(None)):
    

        
    try:
        auth(api_key)
        UPLOAD_DIR = "./sl_faceswap/data/parsingvid/"
        filename = "UPLOAD_FILE.mp4"
        content =await file.read()
        finalname = UPLOAD_DIR + "target/" + filename
        with open(os.path.join(UPLOAD_DIR + "target", filename), "wb") as fp:
            fp.write(content)
        os.chdir('sl_faceswap')
        with open("type.txt", "w") as type_file:
          type_file.write(type)

        os.system('python sl_video_voice.py')
        os.chdir('../')

        with open("./sl_faceswap/data/parsingvid/result/PARSE_OUTPUT_WITH_AUDIO.mp4", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
            return JSONResponse(content={"msg": "success", "video_data": text}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)


@app.post("/parsingImgVid2")
async def parsingImgVid2(source: UploadFile = File(...), target: UploadFile = File(...), api_key: str = Header(None)):
    


    try:
        auth(api_key)

        UPLOAD_DIR = "./sl_faceswap/data/parsingvid_2/"
        filename1 = "UPLOAD_IMG_source.png"
        content1 = await source.read()
        finalname1 = UPLOAD_DIR + "source/" + filename1
        with open(os.path.join(UPLOAD_DIR + "source", filename1), "wb") as fp:
            fp.write(content1)
        filename2 = "UPLOAD_FILE.mp4"
        content2 = await target.read()
        finalname2 = UPLOAD_DIR + "target/" + filename2

        with open(os.path.join(UPLOAD_DIR + "target", filename2), "wb") as fp:
            fp.write(content2)
            


        os.chdir('sl_faceswap')
        os.system('python sl_video_voice_2.py')
        os.chdir('../')

        with open("./sl_faceswap/data/parsingvid_2/result/PARSE_OUTPUT_WITH_AUDIO.mp4", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
            return JSONResponse(content={"msg": "success", "video_data": text}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)

@app.post("/parsingImgVid3")
async def parsingImgVid3(source: UploadFile = File(...), type: TypeEnum = Form(...), api_key: str = Header(None)):
    


    try:
        auth(api_key)

        UPLOAD_DIR = "./sl_faceswap/data/parsingvid_2/"

        filename1 = "UPLOAD_IMG_source.png"
        content1 = await source.read()
        finalname1 = UPLOAD_DIR + "source/" + filename1
        with open(os.path.join(UPLOAD_DIR + "source", filename1), "wb") as fp:
            fp.write(content1)
        viddir = '/home/smartlabs/ss/apitest/sl-parsing-api/sl_faceswap/data/parsingvid_2/target/'
        dest_path = os.path.join(viddir, 'UPLOAD_FILE.mp4')
        if type == 'filter1':
            src_path = os.path.join(viddir, 'dr1.mp4')
            shutil.copyfile(src_path, dest_path)
        elif type == 'filter2':
            src_path = os.path.join(viddir, 'dr2.mp4')
            shutil.copyfile(src_path, dest_path)
        elif type == 'filter3':
            src_path = os.path.join(viddir, 'dr3.mp4')
            shutil.copyfile(src_path, dest_path) 
        elif type == 'filter4':
            src_path = os.path.join(viddir, 'dr4.mp4')
            shutil.copyfile(src_path, dest_path) 
        elif type == 'filter5':
            src_path = os.path.join(viddir, 'dr5.mp4')
            shutil.copyfile(src_path, dest_path) 
        elif type == 'filter6':
            src_path = os.path.join(viddir, 'dr6.mp4')
            shutil.copyfile(src_path, dest_path) 
        elif type == 'filter7':
            src_path = os.path.join(viddir, 'dr7.mp4')
            shutil.copyfile(src_path, dest_path) 
        else:
            print("in else moon")
            src_path = os.path.join(viddir, 'dr1.mp4')
            shutil.copyfile(src_path, dest_path)          
                



        os.chdir('sl_faceswap')
        os.system('python sl_video_voice_2.1.py')
        os.chdir('../')
        source_path = "./sl_faceswap/data/parsingvid_2/result/PARSE_OUTPUT_WITH_AUDIO.mp4"
        destination_dir = '/home/smartlabs/apitest/newweb_test/static'
        destination_path = os.path.join(destination_dir, 'PARSE_OUTPUT_WITH_AUDIO.mp4')
        if os.path.exists(destination_path):
            backup_path = destination_path + "_backup_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".mp4"
            shutil.copyfile(destination_path, backup_path)
            print(f"Backup of the existing file created at: {backup_path}")
            
        shutil.copy(source_path, destination_path)
        print(f"File moved and overwritten if already existed at: {destination_path}")
        suc_value=''

        try:
            with open("sl_faceswap/suc.json", "r") as file:
                data = json.load(file)
                suc_value = data.get("SUC", "Error: SUC key not found")
        except FileNotFoundError:
            suc_value = "Error: File not found"

        with open("./sl_faceswap/data/parsingvid_2/result/PARSE_OUTPUT_WITH_AUDIO.mp4", "rb") as fh:
            text = base64.b64encode(fh.read()).decode('ascii')
            return JSONResponse(content={"msg": "success", "video_data": text, "SUC": suc_value}, status_code=200)
    except Exception as e:
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api')
        logger.error(e)
        return JSONResponse(content={"msg": str(e)}, status_code=500)



def base64_string(output_img):
    oImage = Image.fromarray(output_img.astype('uint8'))
    in_mem_file = io.BytesIO()
    oImage.save(in_mem_file, format="PNG")
    base64_encoded_result_bytes = base64.b64encode(in_mem_file.getvalue())
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    return base64_encoded_result_str

def base64Stringmp4(output_img):
    oImage = Image.fromarray(output_img.astype('uint8'))
    in_mem_file = io.BytesIO()
    oImage.save(in_mem_file, format="mp4")
    base64_encoded_result_bytes = base64.b64encode(in_mem_file.getvalue())
    base64_encoded_result_str = base64_encoded_result_bytes.decode('ascii')
    return base64_encoded_result_str