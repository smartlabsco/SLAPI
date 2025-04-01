import paddle
import cv2
import numpy as np
import os
import glob
import random
from typing import List, Dict, Tuple, Optional
import json

from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel

class FaceswapProcessor:
    def __init__(self, base_path: str = './'):
        """
        FaceswapProcessor 초기화
        Args:
            base_path: 모델 체크포인트와 소스 이미지가 있는 기본 경로
        """
        # 기본 경로 설정
        self.base_path = base_path
        
        # GPU 설정
        paddle.set_device("gpu:1")
        
        # 모델 초기화
        self.faceswap_model = FaceSwap(use_gpu=True)
        self.id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        
        # 체크포인트 로드
        arcface_path = os.path.join(base_path, 'checkpoints/arcface.pdparams')
        weight_path = os.path.join(base_path, 'checkpoints/MobileFaceSwap_224.pdparams')
        
        print(f"Loading arcface from: {arcface_path}")
        print(f"Loading weights from: {weight_path}")
        
        self.id_net.set_dict(paddle.load(arcface_path))
        self.id_net.eval()
        self.weight = paddle.load(weight_path)
        
        # Landmark 모델 초기화
        self.landmark_model = LandmarkModel(name='landmarks')
        self.landmark_model.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))

        # 필터 매핑 정의
        self.filter_mapping = {
            ('백인', '남성'): 'filter1',
            ('백인', '여성'): 'filter2',
            ('황인', '남성'): 'filter3',
            ('황인', '여성'): 'filter4',
            ('흑인', '남성'): 'filter5',
            ('흑인', '여성'): 'filter6',
        }

    def get_id_emb_from_image(self, id_img: np.ndarray) -> Tuple[paddle.Tensor, paddle.Tensor]:
        """이미지에서 임베딩 추출"""
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = self.id_net(id_img)
        id_emb = l2_norm(id_emb)
        return id_emb, id_feature

    def chooseRandomImage(self, filter_type: str) -> str:
        """필터 타입에 따른 랜덤 소스 이미지 선택"""
        source_paths = {
            "filter1": ["data/source_style/24_test_style/white_m/*.png", "data/source_style/24_test_style/white_m/*.jpg"],
            "filter2": ["data/source_style/24_test_style/white_w/*.png", "data/source_style/24_test_style/white_w/*.jpg"],
            "filter3": ["data/source_style/24_test_style/yellow_m/*.png", "data/source_style/24_test_style/yellow_m/*.jpg"],
            "filter4": ["data/source_style/24_test_style/yellow_w/*.png", "data/source_style/24_test_style/yellow_w/*.jpg"],
            "filter5": ["data/source_style/24_test_style/black_m/*.png", "data/source_style/24_test_style/black_m/*.jpg"],
            "filter6": ["data/source_style/24_test_style/black_w/*.png", "data/source_style/24_test_style/black_w/*.jpg"]
        }
        
        paths = source_paths.get(filter_type, ["data/source/*.png", "data/source/*.jpg"])
        full_paths = []
        
        for pattern in paths:
            full_path = os.path.join(self.base_path, pattern)
            full_paths.extend(glob.glob(full_path))
        
        if not full_paths:
            raise ValueError(f"No source images found for filter type: {filter_type}")
            
        while True:
            source_img_random = random.choice(full_paths)
            print(f"Selected source image: {source_img_random}")
            if not source_img_random.endswith('_aligned.png'):
                return source_img_random

    def faces_align(self, image: np.ndarray, image_size: int = 224) -> List[Tuple[np.ndarray, np.ndarray]]:
        """이미지에서 얼굴 정렬"""
        aligned_imgs = []
        landmarks = self.landmark_model.gets(image)
        
        for landmark in landmarks:
            if landmark is not None:
                aligned_img, transform_matrix = align_img(image, landmark, image_size)
                aligned_imgs.append([aligned_img, transform_matrix])
                
        return aligned_imgs

    def process_image(self, image: np.ndarray, filters: List[Dict]) -> np.ndarray:
        """
        이미지 처리 메인 함수
        Args:
            image: 입력 이미지
            filters: 각 얼굴에 적용할 필터 정보 리스트
        Returns:
            처리된 이미지
        """
        # 얼굴 정렬
        target_aligned_images = self.faces_align(image)
        
        if not target_aligned_images:
            raise ValueError("No faces detected in target image")
        
        if len(target_aligned_images) != len(filters):
            raise ValueError(f"Number of detected faces ({len(target_aligned_images)}) "
                           f"doesn't match number of filters ({len(filters)})")

        # 결과 이미지 초기화
        result_img = image.copy()

        # 각 얼굴에 대해 처리
        for idx, (target_face, filter_info) in enumerate(zip(target_aligned_images, filters)):
            try:
                # consent가 False면 해당 얼굴 스킵
                if not filter_info.get('consent', True):
                    print(f"Skipping face {idx} (consent: False)")
                    continue

                # 필터 타입 결정
                filter_type = self.filter_mapping.get(
                    (filter_info['skinTone'], filter_info['gender']),
                    'filter1'
                )

                # 소스 이미지 선택 및 처리
                source_path = self.chooseRandomImage(filter_type)
                source_img = cv2.imread(source_path)
                if source_img is None:
                    print(f"Failed to load source image: {source_path}")
                    continue
                    
                source_aligned_images = self.faces_align(source_img)
                if not source_aligned_images:
                    print(f"No face detected in source image: {source_path}")
                    continue

                # 임베딩 추출 및 모델 설정
                id_emb, id_feature = self.get_id_emb_from_image(source_aligned_images[0][0])
                self.faceswap_model.set_model_param(id_emb, id_feature, model_weight=self.weight)
                self.faceswap_model.eval()

                # 얼굴 교체 수행
                aligned_img, back_matrix = target_face
                att_img = cv2paddle(aligned_img)
                res, mask = self.faceswap_model(att_img)
                res = paddle2cv(res)
                mask = np.transpose(mask[0].numpy(), (1, 2, 0))
                result_img = dealign(res, result_img, back_matrix, mask)

            except Exception as e:
                print(f"Error processing face {idx}: {str(e)}")
                continue

        return result_img

    def process_single_image(self, image_path: str, filters: List[Dict]) -> np.ndarray:
        """
        단일 이미지 파일 처리 (테스트용)
        Args:
            image_path: 처리할 이미지 경로
            filters: 필터 정보
        Returns:
            처리된 이미지
        """
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
            
        # 이미지 처리
        return self.process_image(image, filters)

if __name__ == "__main__":
    # 테스트 코드
    processor = FaceswapParser()
    
    # 테스트 이미지와 필터 설정
    test_image = "test.png"
    test_filters = [
        {"consent": True, "skinTone": "황인", "gender": "남성"},
        {"consent": True, "skinTone": "백인", "gender": "여성"}
    ]
    
    # 이미지 처리
    try:
        result = processor.process_single_image(test_image, test_filters)
        cv2.imwrite("result.png", result)
        print("Processing completed successfully!")
    except Exception as e:
        print(f"Error during processing: {str(e)}")