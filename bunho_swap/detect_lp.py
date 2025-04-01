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
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

class LPSwapParser:
    def __init__(self):
        self.args = None
        self.device = None
        self.detection_network = None
        self.annotate_network = None
        self.transform = None

    def chooseRandomImage(self, type):
        if type == "filter1":
            file_path_type = ["fake_licences/lp_style/1_normal_new_lp/*.png", "fake_licences/lp_style/1_normal_new_lp/*.jpg"]
        elif type == "filter2":
            file_path_type = ["fake_licences/lp_style/2_normal_old_lp/*.png", "fake_licences/lp_style/2_normal_old_lp/*.jpg"]
        elif type == "filter3":
            file_path_type = ["fake_licences/lp_style/3_ev_lp/*.png", "fake_licences/lp_style/3_ev_lp/*.jpg"]
        elif type == "filter4":
            file_path_type = ["fake_licences/lp_style/4_business_lp/*.png", "fake_licences/lp_style/4_business_lp/*.jpg"]
        elif type == "filter5":
            file_path_type = ["fake_licences/lp_style/5_co_lp/*.png", "fake_licences/lp_style/5_co_lp/*.jpg"]
        elif type == "filter6":
            file_path_type = ["fake_licences/lp_style/6_diplomatic_lp/*.png", "fake_licences/lp_style/6_diplomatic_lp/*.jpg"]
        elif type == "filter7":
            file_path_type = ["fake_licences/lp_style/7_heavyeq_lp/*.png", "fake_licences/lp_style/7_heavyeq_lp/*.jpg"]
        else:
            file_path_type = ["fake_licences/lp_final/*.png", "fake_licences/lp_final/*.jpg"]
        
        # 해당 경로에서 랜덤으로 이미지 선택
        images = glob.glob(file_path_type[0]) + glob.glob(file_path_type[1])
        source_img_random = random.choice(images)
        return source_img_random

    def initialize(self, cfg_dir, useGPU=True):
        """Initialize networks and configurations"""
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        config = configparser.RawConfigParser()
        config.read(cfg_dir)

        basic_config = config["basic_config"]

        # Weight Files
        args.detection_weight_file = basic_config["detection_weight_file"]
        args.annotate_weight_file = basic_config["annotate_weight_file"]

        # Input Data File
        args.source = basic_config["source"]

        # GPU Number
        args.gpu_num = basic_config["gpu_num"]

        # Detection Parameters
        args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
        args.detect_save_library = basic_config.getboolean('detect_save_library')
        args.data = basic_config.get("data", 'detection/data/AD.yaml')
        args.half = basic_config.getboolean('half')
        
        imgsz = int(basic_config.get("detect_imgsz", "640"))
        args.detect_imgsz = [imgsz]
        
        args.conf_thres = float(basic_config.get("conf_thres", "0.9"))
        args.iou_thres = float(basic_config.get("iou_thres", "0.45"))
        args.max_det = int(basic_config.get("max_det", "1000"))

        # Output Directory & Result save parameters
        args.output_dir = basic_config["output_dir"]
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        args.result_savefile = basic_config.getboolean('result_savefile')
        args.save_detect_result = basic_config.getboolean('save_detect_result')
        args.save_recog_result = basic_config.getboolean('save_recog_result')
        args.hide_labels = basic_config.getboolean('hide_labels')
        args.hide_conf = basic_config.getboolean('hide_conf')
        args.save_conf = basic_config.getboolean('save_conf')

        # Set up GPU
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
        device = torch.device("cuda:0" if useGPU else "cpu")

        # Build Detection Network
        args.detect_weights = args.detection_weight_file
        detection_network, imgsz, stride, names, pt = build_detect_model(args, device)
        annotate_network = LPDetectionNet(args)

        # Add network parameters to args
        args.pt = pt
        args.stride = stride
        args.imgsz = imgsz
        args.names = names

        # Load networks
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
        """Detect license plates in the image"""
        img_tensor = 255 * img_tensor.permute(1, 2, 0)
        detect_preds = do_detect(self.args, self.detection_network, img_tensor, self.args.imgsz, self.args.stride, auto=True)
        return detect_preds

    def file_to_torchtensor(self, imgname):
        """Convert image file to torch tensor"""
        img_mat = cv2.cvtColor(cv2.imread(imgname), cv2.COLOR_BGR2RGB)
        img_tensor = self.mat_to_torchtensor(img_mat)
        return (img_mat, img_tensor)

    def mat_to_torchtensor(self, img_mat):
        """Convert numpy matrix to torch tensor"""
        img_tensor = self.transform(img_mat)
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def extend_bbox(self, bboxes, img_mat):
        """Extend bounding boxes"""
        H, W, _ = img_mat.shape
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
        """Detect corner points of license plates"""
        transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize([int(128), int(256)]),
        ])    

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

            h, w, c = img_crop.shape
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
        """Swap license plates in the image"""
        ref_img = cv2.cvtColor(cv2.imread(ref_img), cv2.COLOR_BGR2RGB)
        H, W, _ = img_mat.shape
        masked_img_mat = img_mat.copy()

        for point_pred in point_preds:
            points = np.array([
                [point_pred[0], point_pred[1]],  # 좌상단
                [point_pred[6], point_pred[7]],  # 우상단
                [point_pred[4], point_pred[5]],  # 우하단
                [point_pred[2], point_pred[3]]   # 좌하단
            ], dtype=np.int32)

            x, y, w, h = cv2.boundingRect(points)
            
            # 밝기 조정
            ref_img_adjusted = self.match_illumination(ref_img, img_mat, (x, y, w, h))
            
            # 색상 조정
            ref_img_adjusted = self.apply_color_transfer(img_mat[y:y+h, x:x+w], ref_img_adjusted, strength=0.3)
            
            ref_img_resized = cv2.resize(ref_img_adjusted, (w, h))

            # 마스크 생성
            center = np.mean(points, axis=0)
            points_shrunk = points + (center - points) * 0.05
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.fillConvexPoly(mask, points_shrunk.astype(np.int32), 255)
            mask_3channel = cv2.merge([mask, mask, mask])

            # 원근 변환
            src_points = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype=np.float32)
            dst_points = points.astype(np.float32)
            perspective_transform = cv2.getPerspectiveTransform(src_points, dst_points)
            warped = cv2.warpPerspective(ref_img_resized, perspective_transform, (W, H))

            # 엣지 블렌딩
            warped = self.smooth_edges(warped, mask, kernel_size=3, blend_factor=0.5)

            # 합성
            masked_img_mat = cv2.bitwise_and(masked_img_mat, cv2.bitwise_not(mask_3channel))
            masked_img_mat = cv2.bitwise_or(masked_img_mat, cv2.bitwise_and(warped, mask_3channel))

        return masked_img_mat

    def match_illumination(self, source_img, target_img, bbox):
        """Match illumination between source and target images"""
        x, y, w, h = bbox
        surrounding_w = int(w * 2)
        surrounding_h = int(h * 2)
        
        sx = max(0, x - (surrounding_w - w) // 2)
        sy = max(0, y - (surrounding_h - h) // 2)
        ex = min(target_img.shape[1], sx + surrounding_w)
        ey = min(target_img.shape[0], sy + surrounding_h)
        
        surrounding_area = target_img[sy:ey, sx:ex]
        
        src_lab = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab)
        surrounding_lab = cv2.cvtColor(surrounding_area, cv2.COLOR_BGR2Lab)
        
        l_mean_surrounding = np.mean(surrounding_lab[:, :, 0])
        l_std_surrounding = np.std(surrounding_lab[:, :, 0])
        
        l_mean_src = np.mean(src_lab[:, :, 0])
        l_std_src = np.std(src_lab[:, :, 0])
        
        adaptation_factor = min(max(l_std_surrounding / l_std_src, 0.5), 2.0)
        
        l, a, b = cv2.split(src_lab)
        l = l.astype('float64')
        l = (l - l_mean_src) * adaptation_factor + l_mean_surrounding
        
        adjustment_strength = 0.7
        l = l * adjustment_strength + l_mean_src * (1 - adjustment_strength)
        
        l = np.clip(l, 0, 255).astype('uint8')
        adjusted_lab = cv2.merge((l, a, b))
        
        matched_img = cv2.cvtColor(adjusted_lab, cv2.COLOR_Lab2BGR)
        return matched_img

    def apply_color_transfer(self, source, target, strength=0.5):
        """Apply color transfer between images"""
        source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
        target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype("float32")

        source_mean, source_std = [], []
        target_mean, target_std = [], []

        for i in range(3):
            source_mean.append(np.mean(source[:,:,i]))
            source_std.append(np.std(source[:,:,i]))
            target_mean.append(np.mean(target[:,:,i]))
            target_std.append(np.std(target[:,:,i]))

        for i in range(3):
            target[:,:,i] = ((target[:,:,i] - target_mean[i]) * 
                            ((source_std[i] / target_std[i]) ** strength)) + \
                            (target_mean[i] * (1 - strength) + source_mean[i] * strength)

        target = np.clip(target, 0, 255).astype("uint8")
        target = cv2.cvtColor(target, cv2.COLOR_LAB2BGR)
        return target

    def smooth_edges(self, img, mask, kernel_size=3, blend_factor=0.3):
        """Smooth edges of the swapped region"""
        mask_float = mask.astype(float) / 255.0
        mask_blurred = cv2.GaussianBlur(mask_float, (kernel_size, kernel_size), 0)
        mask_blurred = np.stack([mask_blurred] * 3, axis=-1)
        img_blurred = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        blend_mask = mask_blurred * blend_factor
        img_smoothed = img * (1 - blend_mask) + img_blurred * blend_mask
        return img_smoothed.astype(np.uint8)

    def parsing(self, target_img_path, type, merge_result=True, need_align=True):
            """Main parsing and license plate swap function"""
            try:
                # Set target image path
                target_img_path = './data/target/UPLOAD_IMG.png'
                
                # Load and process target image
                img_mat, img_tensor = self.file_to_torchtensor(target_img_path)
                
                # Detect license plates
                bbox = self.detect(img_tensor, img_mat)
                extend_bbox = self.extend_bbox(bbox, img_mat)
                point_preds = self.point_detect(img_mat, extend_bbox)

                # Swap each detected license plate with different source images
                origin_img_mat = img_mat.copy()
                for i, point_pred in enumerate(point_preds):
                    # 각 번호판마다 새로운 source 이미지 선택
                    source_img = self.chooseRandomImage(type)
                    print(f"License plate {i+1} using source image: {source_img}")
                    
                    current_point_pred = [point_pred]
                    origin_img_mat = self.swap_lp(
                        origin_img_mat if i == 0 else origin_img_mat, 
                        current_point_pred, 
                        source_img
                    )

                # Save result
                os.makedirs('./data/result', exist_ok=True)
                cv2.imwrite(
                    './data/result/PARSE_OUTPUT.png',
                    cv2.cvtColor(origin_img_mat, cv2.COLOR_RGB2BGR)
                )
                print("*********************DONE*********************")

            except Exception as e:
                print(f"Error in parsing: {str(e)}")
                raise

def main():
    """Main execution function"""
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print("CUDA Available:", torch.cuda.is_available())
    
    # Create necessary directories
    os.makedirs('./data/target', exist_ok=True)
    os.makedirs('./data/result', exist_ok=True)
    
    try:
        # Read filter type from type.txt
        with open("type.txt", "r") as type_file:
            type = type_file.read().strip()
        
        # Initialize and run parser
        parser = LPSwapParser()
        parser.initialize('detect.cfg', useGPU=True)
        parser.parsing('../UPLOAD_IMG.png', type)
        
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        raise

if __name__ == '__main__':
    main()