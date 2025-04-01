import paddle
import cv2
import numpy as np
import os
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from tqdm import tqdm
from moviepy.editor import *

class FaceswapParser:
    batch_size = 64

    def __init__(self):
        self.landmarkModel = LandmarkModel(name='landmarks')
        self.landmarkModel.prepare(ctx_id=0, det_thresh=0.6, det_size=(640, 640))
        paddle.set_device("gpu")
        self.id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
        self.id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))
        self.id_net.eval()
        
    def face_align(self, image_path, merge_result=False, image_size=224):
        if os.path.isfile(image_path):
            img_list = [image_path]
        else:
            img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith(('png', 'jpg', 'jpeg'))]
        
        for path in img_list:
            img = cv2.imread(path)
            landmark = self.landmarkModel.get(img)
            if landmark is not None:
                base_path = os.path.splitext(path)[0]
                aligned_img, back_matrix = align_img(img, landmark, image_size)
                cv2.imwrite(f"{base_path}_aligned.png", aligned_img)
                if merge_result:
                    np.save(f"{base_path}_back.npy", back_matrix)

    def get_id_emb(self, id_img):
        id_img = cv2.resize(id_img, (112, 112))
        id_img = cv2paddle(id_img)
        mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
        std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
        id_img = (id_img - mean) / std
        id_emb, id_feature = self.id_net(id_img)
        return l2_norm(id_emb), id_feature


    def process_frames(self, frames, faceswap_model):
        results = []
        num_batches = len(frames) // batch_size + int(len(frames) % batch_size != 0)
        
        for i in range(num_batches):
            batch_frames = frames[i*batch_size: (i+1)*batch_size]
            batch_landmarks = [self.landmarkModel.get(frame) for frame in batch_frames]
            batch_aligned = []
            batch_back_matrices = []
            
            for j, (frame, landmark) in enumerate(zip(batch_frames, batch_landmarks)):
                if landmark is not None:
                    att_img, back_matrix = align_img(frame, landmark)
                    att_img = cv2paddle(att_img)
                    batch_aligned.append(att_img)
                    batch_back_matrices.append(back_matrix)
                else:
                    print('**** No Face Detect Error ****')
                    #batch_aligned.append(cv2paddle(frame))
                    batch_back_matrices.append(None)
                    
            batch_res, batch_mask = faceswap_model(paddle.concat(batch_aligned, axis=0))
            batch_res = [paddle2cv(res) for res in paddle.split(batch_res, len(batch_aligned), axis=0)]
            batch_mask = [np.transpose(mask[0].numpy(), (1, 2, 0)) for mask in paddle.split(batch_mask, len(batch_aligned), axis=0)]
            
            for frame, res, back_matrix, mask in zip(batch_frames, batch_res, batch_back_matrices, batch_mask):
                if back_matrix is not None:
                    results.append(dealign(res, frame, back_matrix, mask))
                else:
                    results.append(frame)
        
        return results


    def process_batch(self, batch_frames, faceswap_model):
        aligned_imgs = []
        back_matrices = []

        for frame in batch_frames:
            landmark = self.landmarkModel.get(frame)
            if landmark is not None:
                att_img, back_matrix = align_img(frame, landmark)
                att_img = cv2paddle(att_img)
                aligned_imgs.append(att_img)
                back_matrices.append(back_matrix)
            else:
                print('**** No Face Detect Error ****')
                #aligned_imgs.append(cv2paddle(frame))
                back_matrices.append(None)

        res_imgs, masks = faceswap_model(paddle.concat(aligned_imgs, axis=0))
        processed_frames = []

        for idx in range(res_imgs.shape[0]):
            # Ensure that the tensor shape is (1, channels, height, width)
            res_img_tensor = res_imgs[idx].unsqueeze(0)
            res = paddle2cv(res_img_tensor)
            mask = np.transpose(masks[idx].numpy(), (1, 2, 0))
            if back_matrices[idx] is not None:  # Ensure we have a back_matrix for this frame
                processed_frame = dealign(res, batch_frames[idx], back_matrices[idx], mask)
            else:
                processed_frame = batch_frames[idx]
            processed_frames.append(processed_frame)

        return processed_frames
    def process_single_frame(self, frame, faceswap_model):
        aligned_img = None
        back_matrix = None

        landmark = self.landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            aligned_img = cv2paddle(att_img)
        else:
            print('**** No Face Detect Error ****')
            return frame

        res_img, mask = faceswap_model(aligned_img)
        #print(f"Shape of res_img[0] before transpose: {res_img[0].shape}")

        # Directly converting the image here
        img = res_img[0].numpy()
        #print(f"Shape of img after converting to numpy: {img.shape}")
        img = np.transpose(img, (1, 2, 0))
        #print(f"Shape of img after transpose: {img.shape}")
        img *= 255
        res = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mask = np.transpose(mask[0].numpy(), (1, 2, 0))
        if back_matrix is not None:
            return dealign(res, frame, back_matrix, mask)
        else:
            return frame

    def pasrsing(self, video, type, need_align=True, batch_size = 1):
        os.chdir('/home/smartlabs/ss/apitest/sl-parsing-api/sl_faceswap')
        source='./data/parsingvid_2/source/UPLOAD_IMG_source.png'
        video='./data/parsingvid_2/target/UPLOAD_FILE.mp4'
        
        if need_align:
            self.face_align(source)
            
        base_path = os.path.splitext(source)[0]
        att_img = cv2.imread(f"{base_path}_aligned.png")
        id_img = cv2.imread(source)
        
        landmark = self.landmarkModel.get(id_img)
        if landmark is None:
            print('**** No Face Detect Error ****')
            return
        aligned_id_img, _ = align_img(id_img, landmark)
        id_emb, id_feature = self.get_id_emb(aligned_id_img)
        
        faceswap_model = FaceSwap(use_gpu=True)
        weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')
        faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
        faceswap_model.eval()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        cap = cv2.VideoCapture(video)
        cap.open(video) 

        # Use original FPS
        original_fps = float(cap.get(cv2.CAP_PROP_FPS))
        videoWriter = cv2.VideoWriter('./data/parsingvid_2/result/PARSE_OUTPUT.mp4', fourcc, original_fps, (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        with tqdm(total=total_frames, desc="Processing frames", ncols=100, file=sys.stdout, dynamic_ncols=True) as pbar:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                processed_frame = self.process_single_frame(frame, faceswap_model)
                videoWriter.write(processed_frame)
                pbar.update(1)

            cap.release()
            videoWriter.release()

        # Extract audio from original video and add to the output
        input_video = VideoFileClip(video)
        audio = input_video.audio
        output_video = VideoFileClip('./data/parsingvid_2/result/PARSE_OUTPUT.mp4')
        output_video_with_audio = output_video.set_audio(audio)
        output_video_with_audio.write_videofile('./data/parsingvid_2/result/PARSE_OUTPUT_WITH_AUDIO.mp4', codec='libx264')
        print("*********************DONE*********************")
        os.chdir('../')

if __name__ == '__main__':
    with open("type.txt", "r") as type_file:
        type = type_file.read().strip()
    parser = FaceswapParser()
    parser.pasrsing("../UPLOAD_FILE.mp4", type)

    # Ensure audio synchronization when saving the video
    def save_video_with_audio_sync(self, video_path, processed_frames):
        # Create a video clip from processed frames
        video_clip = ImageSequenceClip(processed_frames, fps=24)  # Assuming fps=24 for the original video

        # Extract audio from the original video
        original_video = VideoFileClip(video_path)
        audio = original_video.audio

        # Set the audio to the processed video
        final_clip = video_clip.set_audio(audio)

        # Save the video with audio synchronization
        output_path = "processed_" + os.path.basename(video_path)
        final_clip.write_videofile(output_path, audio_codec='aac')
    