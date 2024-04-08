import os
import glob
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms 
from torch.nn import functional as F
import cv2


def get_child_folders(directory):
    return [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]

def get_files_with_prefix(directory, prefix):
    return glob.glob(f'{directory}/{prefix}*')

def create_folder(path):
	#inf_output_folder   = os.path.join(os.getcwd() , "inf_out_anime_1109")
	if not os.path.exists(path):
		os.makedirs(path)
	return path


def load_img(img_path , device='cuda'):
	img0 = np.asarray(Image.open (img_path))
	img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)	

	n, c, h, w = img0.shape
	ph = ((h - 1) // 32 + 1) * 32
	pw = ((w - 1) // 32 + 1) * 32
	padding = (0, pw - w, 0, ph - h)
	img0 = F.pad(img0, padding)
	return img0

def save_image_from_np(path , img):
	pil_img = Image.fromarray(img)
	pil_img.save(path)
	pass
def scale_img(img , scale):
	h,w = img.shape[:2]
	new_h = h*scale
	new_w = w*scale

	return  cv2.resize(img, ( int(new_w), int(new_h)), interpolation = cv2.INTER_AREA)

'''
from ignite.engine import *
from ignite.metrics import PSNR ,SSIM
from torchvision import transforms

def eval_step(engine, batch):
    return batch
def get_eval_matric(pred_img , gt_img):
	
	default_evaluator = Engine(eval_step)
	psnr = PSNR(data_range=1.0)
	ssim = SSIM(data_range=1.0)
	psnr.attach(default_evaluator, 'psnr')
	ssim.attach(default_evaluator, 'ssim')

	state = default_evaluator.run([[pred_img, gt_img]])

	return state.metrics['psnr'] , state.metrics['ssim']

	pass
'''

#==========================================
#          Image sequence to video
#==========================================
import re
def sort_key(s):
    # Extract the numbers from the filename
    numbers = re.findall(r'\d+', s)
    # Convert the numbers to integers and return as a tuple
    return tuple(map(int, numbers))

def seq_to_video(in_folder , out_folder, pre_fix="" , video_name ="vid", fps = 30):
    write_path = create_folder(out_folder)
    all_img_path =  get_files_with_prefix( in_folder , pre_fix)    
    #all_img_path = sorted(all_img_path,key=lambda x: os.path.basename(x[:-7]).lower())
    all_img_path = sorted(all_img_path,key=sort_key)
   
    img =  np.asarray(Image.open (all_img_path[0]))
    height, width, layers = img.shape
    
    video_path = os.path.join(out_folder , video_name )
    print("writing...", video_path)
    
    fourcc =  cv2.VideoWriter_fourcc(*'MJPG')
    video = cv2.VideoWriter(video_path+'.avi',fourcc, fps, (width, height), isColor=True)
    
    #for image in all_img_path:
    for i  in range(len(all_img_path)):
        image = all_img_path[i]        
        img =  (np.asarray(Image.open (image)))             
        img =  cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        video.write(img)
        
    cv2.destroyAllWindows()
    video.release()    

    pass

def warp_flow(flow, img1=None, img2=None, interpolation=cv2.INTER_LINEAR):
    """Use remap to warp flow, generating a new image. 
Args:
    flow (np.ndarray): flow
    img1 (np.ndarray, optional): previous frame
    img2 (np.ndarray, optional): next frame
Returns:
    warped image
If img1 is input, the output will be img2_warped, but there will be multiple pixels corresponding to a single pixel, resulting in sparse holes. 
If img2 is input, the output will be img1_warped, and there will be no sparse holes. The latter approach is preferred.
    """
    h, w, _ = flow.shape
    remap_flow = flow.transpose(2, 0, 1)
    remap_xy = np.float32(np.mgrid[:h, :w][::-1])
    if img1 is not None:
        uv_new = (remap_xy + remap_flow).round().astype(np.int32)
        mask = (uv_new[0] >= 0) & (uv_new[1] >= 0) & (uv_new[0] < w) & (uv_new[1] < h)
        uv_new_ = uv_new[:, mask]
        remap_xy[:, uv_new_[1], uv_new_[0]] = remap_xy[:, mask]
        remap_x, remap_y = remap_xy
        img2_warped = cv2.remap(img1, remap_x, remap_y, interpolation)
        mask_remaped = np.zeros((h, w), np.bool8)
        mask_remaped[uv_new_[1], uv_new_[0]] = True
        img2_warped[~mask_remaped] = 0
        return img2_warped
    elif img2 is not None:
        remap_x, remap_y = np.float32(remap_xy + remap_flow)
        return cv2.remap(img2, remap_x, remap_y, interpolation)
    
