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

class FaceswapParser:
    def chooseRandomImage(self, type):
        # 필터 타입에 따라 이미지 경로 선택
        if type == "filter1":
            file_path_type = ["data/source_style/24_test_style/white_m/*.png", "data/source_style/24_test_style/white_m/**.jpg"]
        elif type == "filter2":
            file_path_type = ["data/source_style/24_test_style/white_w/*.png", "data/source_style/24_test_style/white_w/*.jpg"]
        elif type == "filter3":
            file_path_type = ["data/source_style/24_test_style/yellow_m/*.png", "data/source_style/24_test_style/yellow_m/*.jpg"]
        elif type == "filter4":
            file_path_type = ["data/source_style/24_test_style/yellow_w/*.png", "data/source_style/24_test_style/yellow_w/*.jpg"]
        elif type == "filter5":
            file_path_type = ["data/source_style/24_test_style/black_m/*.png", "data/source_style/24_test_style/black_m/*.jpg"]
        elif type == "filter6":
            file_path_type = ["data/source_style/24_test_style/black_w/*.png", "data/source_style/24_test_style/black_w/*.jpg"]
        else:
            file_path_type = ["data/source/*.png", "data/source/*.jpg"]
        
        # 해당 경로에서 랜덤으로 이미지 선택
        images = glob.glob(file_path_type[0]) + glob.glob(file_path_type[1])
        while True:
            source_img_random = random.choice(images)
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

    def get_id_emb_from_image(self, id_net, id_img):
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature

    def faces_align(self, landmarkModel, image_path, image_size=224):
        aligned_imgs = []
        if os.path.isfile(image_path):
            img_list = [image_path]
        else:
            img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]
        for path in img_list:
            img = cv2.imread(path)
            landmarks = landmarkModel.gets(img)
            for landmark in landmarks:
                if landmark is not None:
                    aligned_img, back_matrix = align_img(img, landmark, image_size)
                    aligned_imgs.append([aligned_img, back_matrix])
        return aligned_imgs

    def parsing(self, image, type, merge_result=True, need_align=True):
        # 메인 파싱 및 얼굴 교체 함수
        image = './data/parsingimg_1/target/UPLOAD_IMG.png'
        paddle.set_device("gpu:2")
        faceswap_model = FaceSwap(use_gpu=True)
        id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        id_net.eval()
        weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
        
        source_img = self.chooseRandomImage(type)
        print("********source_img:::: ", source_img)

        if need_align:
            landmarkModel = LandmarkModel(name='landmarks')
            landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
            source_aligned_images = self.faces_align(landmarkModel, source_img)
            target_aligned_images = self.faces_align(landmarkModel, image, image_size=224)

        origin_att_img = cv2.imread(image)

        for idx, target_aligned_image in enumerate(target_aligned_images):
            id_emb, id_feature = self.get_id_emb_from_image(id_net, source_aligned_images[idx % len(source_aligned_images)][0])
            faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
            faceswap_model.eval()
            
            att_img = cv2paddle(target_aligned_image[0])
            res, mask = faceswap_model(att_img)
            res = paddle2cv(res)

            if merge_result:
                back_matrix = target_aligned_images[idx % len(target_aligned_images)][1]
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                origin_att_img = dealign(res, origin_att_img, back_matrix, mask)

        cv2.imwrite('./data/parsingimg_1/result/PARSE_OUTPUT.png', origin_att_img)
        print("*********************DONE*********************")
        os.chdir('../')

if __name__ == '__main__':
    with open("type.txt", "r") as type_file:
        type = type_file.read().strip()
    parser = FaceswapParser()
    parser.parsing('../UPLOAD_IMG.png', type)