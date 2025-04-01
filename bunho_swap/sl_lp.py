from __future__ import annotations
import torch
import torchvision.transforms as transforms

import configparser
import argparse
import os
import sys
import cv2
from pathlib import Path
import numpy as np
import glob
import random
from datetime import datetime

from annotate.model import LPDetectionNet

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.execute import do_detect, build_detect_model

os.chdir(SAVE_CWD)
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

class DetectLP:
    def __init__(self):
        self.args = None
        self.device = None
        self.detection_network = None
        self.transform = None

    def initialize(self, cfg_dir, useGPU=True):

        ################################################
        #   1. Read in parameters from config file     #
        ################################################
        if True:
            parser = argparse.ArgumentParser()

            args = parser.parse_args()

            config = configparser.RawConfigParser()
            config.read(cfg_dir)

            basic_config = config["basic_config"]

            # Weight Files
            args.detection_weight_file = basic_config["detection_weight_file"]
            if not os.path.exists(args.detection_weight_file):
                print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
                #sys.exit(2)

            args.annotate_weight_file = basic_config["annotate_weight_file"]
            if not os.path.exists(args.detection_weight_file):
                print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
                #sys.exit(2)

            # Input Data File
            args.source = basic_config["source"]
            if not os.path.exists(args.source):
                print(">>> NOT Exist INPUT File {0}".format(args.source))

            # GPU Number
            args.gpu_num = basic_config["gpu_num"]
            if args.gpu_num == "" :
                print(">>> NOT Assign GPU Number")
                #sys.exit(2)

            # Detection Parameters
            args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
            if args.infer_imsize_same == "" :
                args.infer_imsize_same = False

            args.detect_save_library = basic_config.getboolean('detect_save_library')
            if args.detect_save_library == "" :
                args.detect_save_library = False

            args.data = basic_config["data"]
            if args.data == "" :
                args.data = 'detection/data/AD.yaml'

            args.half = basic_config.getboolean('half')
            if args.half == "" :
                args.half = False

            imgsz = int(basic_config["detect_imgsz"])
            args.detect_imgsz = [imgsz]
            if args.detect_imgsz == "" :
                args.detect_imgsz = [640]

            args.conf_thres = float(basic_config["conf_thres"])
            if args.conf_thres == "" :
                args.conf_thres = 0.9

            args.iou_thres = float(basic_config["iou_thres"])
            if args.iou_thres == "" :
                args.iou_thres = 0.45

            args.max_det = int(basic_config["max_det"])
            if args.max_det == "" :
                args.max_det = 1000


            # Add Directory & Result save parameters
            args.output_dir = basic_config["output_dir"]
            if not os.path.exists(args.output_dir):
                os.makedirs(args.output_dir)

            args.result_savefile = basic_config.getboolean('result_savefile')
            if args.result_savefile == "" :
                args.result_savefile = False

            args.save_detect_result = basic_config.getboolean('save_detect_result')
            if args.save_detect_result == "" :
                args.save_detect_result = False

            args.save_recog_result = basic_config.getboolean('save_recog_result')
            if args.save_recog_result == "" :
                args.save_recog_result = False

            args.hide_labels = basic_config.getboolean('hide_labels')
            if args.hide_labels == "" :
                args.hide_labels = False

            args.hide_conf = basic_config.getboolean('hide_conf')
            if args.hide_conf == "" :
                args.hide_conf = False

            args.save_conf = basic_config.getboolean('save_conf')
            if args.save_conf == "" :
                args.save_conf = False


            # Other parameters
            args.deidentified_type = basic_config["deidentified_type"]
            if args.deidentified_type == "" :
                args.deidentified_type = 2


        ################################################
        #                2. Set up GPU                 #
        ################################################
        if True:
            os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
            device = torch.device("cuda:0" if useGPU else "cpu")

        ################################################
        #  3. Declare Detection / Recognition Network  #
        ################################################
        if True:
            # Detection Network
            args.detect_weights = args.detection_weight_file
            detection_network, imgsz, stride, names, pt = build_detect_model(args, device)
            annotate_network = LPDetectionNet(args)
            # Add network parameters to args
            args.pt = pt
            args.stride = stride
            args.imgsz = imgsz
            args.names = names  

        ################################################
        #    4. Load Detection / Recognition Network   #
        ################################################
        if True:
            annotate_checkpoint = torch.load(args.annotate_weight_file, map_location=device)
            annotate_network.load_state_dict(annotate_checkpoint['network'])            
            annotate_network.to(device)
            with torch.no_grad():
                detection_network.eval()
                annotate_network.eval()

        self.args = args
        self.device = device
        self.detection_network = detection_network
        self.annotate_network = annotate_network
        self.transform = transforms.ToTensor()

    def detect(self, img_tensor, img_mat):

        img_tensor = 255*img_tensor.permute(1,2,0)

        detect_preds = do_detect(self.args, self.detection_network, img_tensor, self.args.imgsz, self.args.stride, auto=True)

        return detect_preds

    def file_to_torchtensor(self, imgname):
        
        img_mat = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)

        img_tensor = self.mat_to_torchtensor(img_mat)

        return (img_mat, img_tensor)
    
    def mat_to_torchtensor(self, img_mat):
        
        img_tensor = self.transform(img_mat)
        img_tensor = img_tensor.to(self.device)

        return img_tensor

    def extend_bbox(self, bboxes, img_mat):

        H,W,_ = img_mat.shape

        ext_bboxes = bboxes.clone()

        for idx, bbox in enumerate(bboxes):
            cx = (bbox[2] + bbox[0])/2
            cy = (bbox[3] + bbox[1])/2
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]

            new_x1 = max(cx - 0.75*w, 0)
            new_x2 = min(cx + 0.75*w, W)
            new_y1 = max(cy - 0.75*h, 0)
            new_y2 = min(cy + 0.75*h, H)

            ext_bboxes[idx][0] = int(new_x1)
            ext_bboxes[idx][1] = int(new_y1)
            ext_bboxes[idx][2] = int(new_x2)
            ext_bboxes[idx][3] = int(new_y2)
        
        return ext_bboxes

    def point_detect(self, img_mat, extend_bbox):

        transform_img = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Resize([int(128), int(256)]),
            ]
        )    

        extend_bbox = extend_bbox.detach().cpu().numpy()

        f_preds = []

        for bbox in extend_bbox:

            x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

            img_crop = img_mat[y1:y2, x1:x2,:]

            img_crop_t = transform_img(img_crop)
            img_crop_t = img_crop_t.to(self.device)
            img_crop_t = torch.unsqueeze(img_crop_t, dim=0)

            pred = self.annotate_network(img_crop_t)

            pred = pred.detach().cpu().numpy().squeeze()

            h,w,c = img_crop.shape
            pred[0::2] = pred[0::2] * w
            pred[1::2] = pred[1::2] * h

            pred = pred.astype('int32')

            f_x1 = x1 + pred[0]
            f_x2 = x1 + pred[2]
            f_x3 = x1 + pred[4]
            f_x4 = x1 + pred[6]
            f_y1 = y1 + pred[1]
            f_y2 = y1 + pred[3]
            f_y3 = y1 + pred[5]
            f_y4 = y1 + pred[7]
            
            f_preds.append([f_x1, f_y1, f_x2, f_y2, f_x3, f_y3, f_x4, f_y4])

        return f_preds

    def swap_lp(self, img_mat, point_preds, ref_img):
        ref_img = cv2.cvtColor(cv2.imread(ref_img), cv2.COLOR_BGR2RGB)

        H, W, _ = img_mat.shape
        ref_img = cv2.resize(ref_img, (W,H))

        result = img_mat.copy()
        mask_sum = np.zeros_like(img_mat[:, :, 0])

        for point_pred in point_preds:
            point_matrix = np.float32([[0,0], [W, 0], [0,H], [W,H]])
            
            lu = [point_pred[0], point_pred[1]]
            ru = [point_pred[6], point_pred[7]]
            ld = [point_pred[2], point_pred[3]]
            rd = [point_pred[4], point_pred[5]]
            print("hihihi")

            converted_points = np.float32([lu, ru, ld, rd])

            perspective_transform = cv2.getPerspectiveTransform(point_matrix,converted_points)

            warped = cv2.warpPerspective(ref_img,perspective_transform,(W,H))

            mask = (np.mean(warped, axis=2) > 0).astype(np.uint8)
            mask = self.smooth_edges(mask, mask)
            mask = np.stack([mask, mask, mask], axis=-1)

            warped = self.add_texture_to_plate(warped, img_mat, mask)
            warped = self.add_noise_texture(warped, intensity=5)

            result = np.where(mask == 1, warped, result)
            mask_sum += mask[:, :, 0]

        mask_sum_binary = (mask_sum > 0).astype(np.uint8)
        mask_sum_binary = np.stack([mask_sum_binary, mask_sum_binary, mask_sum_binary], axis=-1)

        result = np.where(mask_sum_binary == 1, result, img_mat)

        return result
    
    


    def color_correction(self, source_img, target_img, mask):
        source_masked = source_img * mask
        target_masked = target_img * mask

        source_hist = cv2.calcHist([source_masked], [0, 1, 2], mask[:,:,0], [256, 256, 256], [0, 256, 0, 256, 0, 256])
        target_hist = cv2.calcHist([target_masked], [0, 1, 2], mask[:,:,0], [256, 256, 256], [0, 256, 0, 256, 0, 256])

        source_hist_cdf = source_hist.cumsum()
        target_hist_cdf = target_hist.cumsum()

        lut = np.zeros((3, 256), dtype=np.float32)  # float32 타입으로 변경
        for i in range(3):
            lut[i] = np.interp(source_hist_cdf[i], target_hist_cdf[i], np.arange(256))

        lut = np.clip(lut, 0, 255).astype(np.uint8)  # 값을 0과 255 사이로 클리핑하고 uint8로 변환

        corrected_img = cv2.LUT(source_img, lut)
        return corrected_img
    
    def poisson_blending(self, source_img, target_img, mask):
        mask = mask[:,:,0]  # Convert mask to single channel
        blended_img = cv2.seamlessClone(source_img, target_img, mask, (target_img.shape[1]//2, target_img.shape[0]//2), cv2.NORMAL_CLONE)
        return blended_img
    
    def add_noise_texture(self, img, intensity=5):
        # 노이즈를 생성합니다.
        noise = np.random.normal(loc=0, scale=intensity, size=img.shape).astype(np.uint8)

        # 노이즈를 이미지에 추가합니다.
        img_noised = cv2.add(img, noise, dtype=cv2.CV_8UC3)

        return img_noised

   


    def save_cropped_images(self, img_mat, point_preds, base_filename):
        """
        This function saves the cropped images based on point_preds coordinates.

        Parameters:
        img_mat (numpy array): The original image matrix.
        point_preds (list of lists): The list of point coordinates.
        base_filename (str): The base filename to use for saving cropped images.
        """
        for idx, point_pred in enumerate(point_preds):
            # Convert the points to numpy array and reshape
            points = np.array(point_pred).reshape((4, 2))
            # Create an all-zero mask with the same dimensions as the original image
            mask = np.zeros(img_mat.shape, dtype=np.uint8)
            # Fill the detected polygon with white in the black mask
            cv2.fillConvexPoly(mask, points, (255, 255, 255))
            # Bitwise the original image and the mask
            result = cv2.bitwise_and(img_mat, mask)
            # Get the bounding box around the detected polygon
            x,y,w,h = cv2.boundingRect(points)
            # Crop the result using the bounding box coordinates
            cropped_result = result[y:y+h, x:x+w]
            # Save the cropped image
            cv2.imwrite(os.path.join('onebon_bunho', f"{base_filename}_{idx}.png"), cropped_result)

    def match_illumination(self, source_img, target_img):
        # YCrCb 색공간에서 Y 채널(밝기)만 히스토그램 평활화를 적용합니다.
        ycrcb = cv2.cvtColor(target_img, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        target_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

        # 원본 이미지와 타겟 이미지의 밝기 차이를 계산합니다.
        src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab)
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2Lab)
        l_mean_src = np.mean(src_lab[:, :, 0])
        l_mean_target = np.mean(target_lab[:, :, 0])
        l_std_src = np.std(src_lab[:, :, 0])
        l_std_target = np.std(target_lab[:, :, 0])

        # 밝기 차이에 따라 타겟 이미지의 L 채널을 조정합니다.
        l, a, b = cv2.split(target_lab)
        l = l.astype('float64')  # L 채널을 float64 타입으로 변환
        l -= l_mean_target
        l = (l_std_src / l_std_target) * l
        l += l_mean_src
        l = np.clip(l, 0, 255).astype('uint8')  # 결과를 0~255 범위로 제한하고 uint8로 다시 변환
        target_lab = cv2.merge((l, a, b))
        
        # 최종 조정된 이미지를 반환합니다.
        matched_img = cv2.cvtColor(target_lab, cv2.COLOR_Lab2BGR)
        return matched_img

    # 엣지 스무딩 함수
    def smooth_edges(self, img, mask, kernel_size=5):
        # mask를 3채널로 확장합니다.
        mask_blurred = cv2.GaussianBlur(mask, (kernel_size, kernel_size), 0)
        mask_blurred = np.stack([mask_blurred]*3, axis=-1)  # 3차원으로 확장

        # 이미지에 블러를 적용합니다.
        img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

        # 블러된 마스크를 사용하여 엣지를 스무딩합니다.
        img_masked = (img * (1 - mask_blurred) + img_blurred * mask_blurred).astype(np.uint8)
        
        return img_masked

    # 텍스처 매칭 함수
    def add_texture_to_plate(self, img, texture_img, mask):
        # 텍스처 이미지를 번호판 이미지 크기에 맞게 조정합니다.
        texture_resized = cv2.resize(texture_img, (img.shape[1], img.shape[0]))
        
        # 두 이미지가 동일한 데이터 타입을 갖도록 합니다. 여기서는 uint8을 사용합니다.
        texture_resized = texture_resized.astype(np.uint8)
        img = img.astype(np.uint8)
        
        # 합성을 위해 텍스처 이미지를 번호판 이미지와 같은 크기로 조정합니다.
        texture_masked = texture_resized * mask
        texture_masked = texture_masked.astype(np.uint8)  # 데이터 타입을 uint8로 변환합니다.

        # 두 이미지를 가중치를 두어 합성합니다.
        img_textured = cv2.addWeighted(img, 1, texture_masked, 0.5, 0)
        return img_textured
 

import os
import glob
import cv2
from detect_lp import DetectLP  # 가정: DetectLP는 필요한 모든 메소드를 포함한 클래스입니다.

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("caon I use GPU?")
    print(torch.cuda.is_available())
    detectlp = DetectLP()
    detectlp.initialize('detect.cfg', useGPU=True)

    # 'dataset/test' 경로의 모든 .png 파일에 대해 로직 수행
    test_file_paths = glob.glob('dataset/test/*.png') + glob.glob('dataset/test/*.jpg')
    # 'lp_test/source' 경로의 모든 .png와 .jpg 파일
    #주간
    source_file_paths = glob.glob('fake_licences/lp_ori/*.png') + glob.glob('fake_licences/lp_ori/*.jpg')
    #야간
    #source_file_paths = glob.glob('fake_licences/lp_night/*.png') + glob.glob('fake_licences/lp_night/*.jpg')

    # 결과를 저장할 디렉터리 생성
    result_dir = 'result'
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    for file_path in test_file_paths:
        base_filename = os.path.basename(file_path)
        test_filename_without_extension = os.path.splitext(base_filename)[0]

        # 랜덤하게 source 파일 선택
        random_source_file_path = random.choice(source_file_paths)

        # 이미지를 텐서로 변환하고 디텍션 수행
        img_mat, img_tensor = detectlp.file_to_torchtensor(file_path)
        bbox = detectlp.detect(img_tensor, img_mat)
        extend_bbox = detectlp.extend_bbox(bbox, img_mat)
        point_preds = detectlp.point_detect(img_mat, extend_bbox)

        img_swapped = detectlp.swap_lp(img_mat, point_preds, random_source_file_path)

        # 이미지에서 알파 채널 제거
        # img_swapped = detectlp.make_non_transparent(img_swapped)

        # 결과물 저장
        current_time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        result_file_path = os.path.join(result_dir, f"{test_filename_without_extension}_{current_time_str}_result.jpg")
        cv2.imwrite(result_file_path, cv2.cvtColor(img_swapped, cv2.COLOR_RGB2BGR))
