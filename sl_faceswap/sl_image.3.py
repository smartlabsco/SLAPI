from genericpath import isfile
from heapq import merge
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
import shutil
import json
#faceswap parser iamge
class FaceswapParser:

    #Randomly select filter image
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

        
    def get_id_emb(self, id_net, id_img_path):
        id_img = cv2.imread(id_img_path)
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        
        return id_emb, id_feature

    def crop_center(self, img):
        h, w = img.shape[:2]
        new_h, new_w = 2 * h // 3, w
        startx = w // 2 - (new_w // 2)
        starty = h // 2 - (new_h // 2)    
        return img[starty:starty+new_h, startx:startx+new_w], startx, starty, new_w, new_h

    def face_align(self, landmarkModel, image_path, merge_result=False, image_size=224):
        success = False
        if os.path.isfile(image_path):
            img_list = [image_path]
        else:
            img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
        for path in img_list:
            img = cv2.imread(path)
            # 큰 얼굴에 대해 더 넓은 범위에서 랜드마크를 감지하기 위해 이미지 전체를 사용
            landmark = landmarkModel.get(img)  # 변경: 크롭 대신 전체 이미지 사용
            if landmark is not None:
                success= True
                base_path = path.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
                # 큰 얼굴을 정확하게 정렬하기 위해 필요한 경우 image_size를 조정
                aligned_img, back_matrix = align_img(img, landmark, image_size)  # 변경: 크롭된 이미지 대신 전체 이미지 사용
                cv2.imwrite(base_path + '_aligned.png', aligned_img)
                if merge_result:
                    np.save(base_path + '_back.npy', back_matrix)

        return success


    def parsing(self, image, type, merge_result=True, need_align=True):
        image = './data/parsingimg_1/target/UPLOAD_IMG.png'
        # load model
        paddle.set_device("gpu")
        faceswap_model = FaceSwap(use_gpu=True)
        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        id_net.eval()
        weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
        print("my self image name :::: ",self.chooseRandomImage(type))
        source_img = self.chooseRandomImage(type)
        base_path = source_img.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
        id_emb, id_feature = self.get_id_emb(id_net, base_path + '_aligned.png')
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        if need_align:
            landmarkModel = LandmarkModel(name='landmarks')
            landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            self.face_align(landmarkModel, source_img)
            #self.face_align(landmarkModel, image, merge_result=True, image_size=224)

        if os.path.isfile(image):
            img_list = [image]
        else:
            img_list = [os.path.join(image, x) for x in os.listdir(image) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
            
        for img_path in img_list:

            landmarkModel = LandmarkModel(name='landmarks')
            landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            origin_att_img = cv2.imread(img_path)
            h, w = origin_att_img.shape[:2]
            new_h, new_w = 2 * h // 3, w//3
            startx = w // 2 - (new_w // 2)
            starty = h // 2 - (new_h // 2)








            cropped_img = origin_att_img[starty:starty+new_h, startx:startx+new_w]
            cpn= "./data/parsingimg_1/result/PARSE_cropimg.png"
            cv2.imwrite(cpn, cropped_img)
            align_success = self.face_align(landmarkModel, cpn, merge_result=True, image_size=224)
            print(align_success)
            if not align_success:

                data = {"SUC": "01"}  # 저장할 데이터

                with open("suc.json", "w") as file:
                    json.dump(data, file)  # 파일에 데이터 쓰기


                # 얼굴 정렬에 실패한 경우, 원본 이미지를 결과로 저장하고 함수를 종료합니다.
                shutil.copy(image, './data/parsingimg_1/result/PARSE_OUTPUT.png')
                print("Face alignment failed. The original image has been copied as the result.")
                return  # 얼굴 정렬 실패 시, 이후 작업을 중단하고 함수 종료
            

            data = {"SUC": "00"}  # 저장할 데이터

            with open("suc.json", "w") as file:
                json.dump(data, file)  # 파일에 데이터 쓰기






            base_path = cpn.replace('.png', '').replace('.jpg', '').replace('.jpeg', '')
            att_img = cv2.imread(base_path + '_aligned.png')
            att_img = cv2paddle(att_img)

            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)



            if merge_result:

                back_matrix = np.load(base_path + '_back.npy')
                print("back jwapyo ?")
                print(base_path + '_back.npy')
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                res = dealign(res, cropped_img, back_matrix, mask)
                cv2.imwrite('./data/parsingimg_1/result/PARSE_OUTPUT_resres.png', res)
                print("merging")
                # 크롭된 결과를 원본 이미지에 다시 붙여넣습니다.
                origin_att_img[starty:starty+new_h, startx:startx+new_w] = res
            
            cv2.imwrite('./data/parsingimg_1/result/PARSE_OUTPUT.png', origin_att_img)
            print("*********************DONE*********************")
        os.chdir('../')

if __name__ == '__main__':
    with open("type.txt", "r") as type_file:
        type = type_file.read().strip()
    parser = FaceswapParser()
    parser.parsing('../UPLOAD_IMG.png', type)