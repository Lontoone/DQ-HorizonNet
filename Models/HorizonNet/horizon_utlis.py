import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
current_file = os.getcwd()
os.chdir(dname)

import torch
import sys
print(os.getcwd())
sys.path.append('../../')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import gridspec
os.chdir(current_file)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def predict (data , net):    
    imgs = data['image'].to(device)        
    out = net(imgs)    
    return out

def fig_to_img(fig):    
    img = np.asarray(fig.canvas.buffer_rgba())
    return img

def save_model(net, path , epoch =0 , ap = 0):
    state_dict = {
        #'args': args.__dict__,
        'kwargs': {
            'backbone': net.backbone,
            'use_rnn': net.use_rnn,
        },
        'state_dict': net.state_dict(),
        'epoch':epoch,
        'ap':ap
    }
    torch.save(state_dict, path)

def visualize_2d(us, v_tops , v_btms, imgs, u_grad=None  , title =None , do_sig_u =False , polys = None ,  save_path=""):
    out_imgs=[]    
    length =  imgs.shape[0] if torch.is_tensor(imgs) else  len(imgs)        

    for i in range(length):
        if polys  is not None and u_grad is not None:            
            img =visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , u_grad[i] , title ,do_sig_u , polys[i]  , save_path= f"{save_path}/{title}_{i}.jpg")
        elif u_grad is not None:            
            img =visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , u_grad[i] , title ,do_sig_u ,polys, save_path= f"{save_path}/{title}_{i}.jpg" )
        else:
            img = visualize_2d_single(us[i] , v_tops[i] ,v_btms[i] , imgs[i] , None , title , do_sig_u , polys, save_path= f"{save_path}/{title}_{i}.jpg")

        out_imgs.append(img)
    return out_imgs



def visualize_2d_single(us, v_tops , v_btms, imgs, u_grad=None , title=None , do_sig_u =False , poly =None , save_path=""):
    if isinstance(us, torch.Tensor):
        us = us.cpu().detach().numpy().flatten()
        v_tops = v_tops.cpu().detach().numpy().flatten()
        v_btms = v_btms.cpu().detach().numpy().flatten()
    else:    
        us=np.array([u.cpu().detach().numpy() for u in us]).flatten()
        v_tops= np.array([u.cpu().detach().numpy().flatten() for u in v_tops]).flatten()
        v_btms=np.array([u.cpu().detach().numpy().flatten() for u in v_btms]).flatten()
    uvs=[]
    for u, v_t , v_b in zip( us , v_tops ,v_btms):   
        uvs.append( (u , v_t) )
        uvs.append( (u , v_b) )        
        
    img = imgs.permute(1,2,0).cpu().detach().numpy()
    img = np.ascontiguousarray(img)

    if(poly is not None):        
        for doors in poly:            
            for part_door in doors:            
                part_door = np.array(part_door)                
                part_door = part_door.reshape((-1 , 2)) * np.tile(np.array([1024 , 512]) , (part_door.size//2 , 1) )
                part_door = part_door.astype('int32')               
                img =  cv2.polylines(img, [part_door], True, (0,255,0), 2)
        pass
    
    h,w,c = img.shape
    img_size = [w,h]    
    for point in uvs:
        #p = np.float32(point) * img_size % img_size       # clamp to boarder     
        p = np.float32(point) * img_size         
        p = np.int32(p)        
        img = cv2.circle( img, tuple( (p[0] , p[1])), 5,(255,0,0) , thickness= -1)

    # Preview Confidence map
    if u_grad is not None:        
        fig = plt.figure()
        spec = gridspec.GridSpec(ncols=1, nrows=3,)
        fig.tight_layout()        
        if do_sig_u ==True:
            u_grad = torch.sigmoid( u_grad)
        dist_graph = u_grad.repeat((50,1)).cpu().detach().numpy()            
            
        ax0 = fig.add_subplot(spec[0])
        ax0.imshow(dist_graph , cmap="gray" )
        ax0.axis("off")        

        ax0 = fig.add_subplot(spec[1:])
        ax0.imshow(img , aspect='auto' )
        ax0.axis("off")        
        
        if(title is not None):
            fig.suptitle(title)
        if save_path != "":
            plt.savefig(save_path)
            plt.close() 
     
        return fig_to_img(fig)
    else:
     
        return img
    pass


def get_grad_u(u ,_width = 1024 , c = 0.96):    
    u_len = u.shape[0]
    width = _width
    dist = torch.arange(0, width)
    #dist = dist.tile((u.shape[0],1) )            
    dist = dist.repeat((u.shape[0],1) )            
    dist = torch.abs( dist.float() - u.reshape((-1,1))*width )        
    c_dist = c ** dist              
    
    #c_dist[:u_len//2] = torch.max(c_dist[ 0::2 ] , c_dist[ 1::2 ])
    
    return c_dist

def get_grad_u_keep_batch(batch_u , pair =False , width = 1024):
    result =[]
    for u in batch_u:            
        w = u.shape[0]
        if (pair):
            u1 = get_grad_u(u[:w//2].cpu().detach() , width)
            u2 = get_grad_u(u[w//2:].cpu().detach() , width)
            result.append( torch.max(u1,u2))
        else:
            result.append(get_grad_u(u.cpu().detach() , width) )
    
    return result
from scipy import ndimage
def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = ndimage.maximum_filter(signal, size=r, mode='wrap')    
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]


def to_bbox( u_pack , vt_pack , vb_pack ):

    u_flatten  = torch.cat(u_pack)
    vt_flatten  = torch.cat(vt_pack)
    vb_flatten  = torch.cat(vb_pack)

    non_zero_idx = torch.where(u_flatten>0)[0]
    u_flatten = u_flatten[non_zero_idx]
    vt_flatten = vt_flatten[non_zero_idx].reshape(-1 , 2)
    vb_flatten = vb_flatten[non_zero_idx].reshape(-1 , 2)

    vt_flatten = torch.min(vt_flatten , 1)[0]
    vb_flatten = torch.max(vb_flatten , 1)[0]

    bboxes=[]
    for i in range(vt_flatten.shape[0]):
        x1 = u_flatten[2*i]
        x2 = u_flatten[2*i+1]
        y1 = vt_flatten[i]
        y2 = vb_flatten[i]
        bboxes.append((x1,y1,x2,y2))
    bboxes = torch.as_tensor(bboxes).reshape(-1,4)
    return bboxes 
    pass
