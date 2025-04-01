import paddle
import cv2
import numpy as np
import os
import glob
import random
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm
import glob
import random
from moviepy.editor import *  # Add this import


# faceswap parser
class FaceswapParser:

    #필터 이미지를 랜덤으로 선택
    def chooseRandomImage(self, type):
        if type == "filter1":
            file_path_type = ["data/source/asian_m/*.png", "data/source/asian_m/*.jpg"]
        elif type == "filter2":
            file_path_type = ["data/source/asian_g/*.png", "data/source/asian_g/*.jpg"]
        elif type == "filter3":
            file_path_type = ["data/source/western_m/*.png", "data/source/western_m/*.jpg"]
        elif type == "filter4":
            file_path_type = ["data/source/western_g/*.png", "data/source/western_g/*.jpg"]
        elif type == "filter5":
            file_path_type = ["data/source/black_m/*.png", "data/source/black_m/*.jpg"]
        elif type == "filter6":
            file_path_type = ["data/source/black_g/*.png", "data/source/black_g/*.jpg"]
        else:
            file_path_type= ["data/source/*.png", "data/source/*.jpg"]

        images = glob.glob(file_path_type[0]) + glob.glob(file_path_type[1])
        while True:
            source_img_random = random.choice(images)
            # Continue to next iteration if the file ends with '_aligned.png'
            if source_img_random.endswith('_aligned.png'):
                continue
            else:
                return source_img_random


    def get_id_emb(self, id_net, id_img):
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature

    def pasrsing(self, video, type):
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api/sl_faceswap')
        video='./data/parsingvid/target/UPLOAD_FILE.mp4'
        #load model
        print(" video name : ", video)
        paddle.set_device("gpu")
        faceswap_model = FaceSwap(use_gpu=True)
        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        id_net.eval()
        weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
        landmarkModel = LandmarkModel(name='landmarks')
        landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640, 640))
        source_img = self.chooseRandomImage(type)
        print(" source_img : ", source_img)
        id_img = cv2.imread(source_img)

        landmark = landmarkModel.get(id_img)
        if landmarkModel is None:
            print('**** No Face Detect Error ****')
            exit()
        aligned_id_img, _ = align_img(id_img, landmark)
        id_emb, id_feature = self.get_id_emb(id_net, aligned_id_img)
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        #load video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(video)
        cap.open(video) 



        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter('./data/parsingvid/result/PARSE_OUTPUT.mp4', fourcc, original_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
        all_f = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        for i in tqdm(range(int(all_f))):
            ret, frame = cap.read()
            landmark = landmarkModel.get(frame)
            if landmark is not None:
                att_img, back_matrix = align_img(frame, landmark)
                att_img = cv2paddle(att_img)
                res, mask = faceswap_model(att_img)
                res = paddle2cv(res)
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                res = dealign(res, frame, back_matrix, mask)
                frame = res
            else:
                print('**** No Face Detect Error ****')
            videoWriter.write(frame)
        cap.release()
        videoWriter.release()
         # Extract audio from original video
        input_video = VideoFileClip(video)
        audio = input_video.audio
         # Audio to the output video
        output_video = VideoFileClip('./data/parsingvid/result/PARSE_OUTPUT.mp4')
        output_video_with_audio = output_video.set_audio(audio)
        output_video_with_audio.write_videofile('./data/parsingvid/result/PARSE_OUTPUT_WITH_AUDIO.mp4', codec='libx264')
        print("*********************DONE*********************")
        os.chdir('../')

        #return frame

if __name__ == '__main__':
    with open("type.txt", "r") as type_file:
        type = type_file.read().strip()
    parser = FaceswapParser()
    #parser.pasrsing("../feelsong.mp4")   
    parser.pasrsing("../UPLOAD_FILE.mp4", type)
    