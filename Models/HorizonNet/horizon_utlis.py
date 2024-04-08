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
        '''
        plt.show()
        '''
        return fig_to_img(fig)
    else:
        #plt.title(title)
        #plt.imshow(img)
        #plt.show()
        return img
    pass


# encode for fasten training 
def encode(packed_data):
    _esp = 0.000001  # avoid door width = 0 
    packed_data[:,1] = torch.log( 0.5 - packed_data[:,1])  #v_top
    packed_data[:,2] = torch.log( packed_data[:,2] - 0.5)  #v_btm
    packed_data[:,3] = torch.log( packed_data[:,3] + _esp )  #du
    packed_data[:,4] = torch.log( 0.5 - packed_data[:,4] + _esp)  #v_top2
    packed_data[:,5] = torch.log( packed_data[:,5] - 0.5 + _esp)  #v_btm2

    zeros = torch.zeros_like(packed_data)
    is_nan = torch.isnan(packed_data)
    packed_data = torch.where(is_nan , zeros , packed_data)    
    
    return packed_data
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

def uv_to_xyz(u,v):
    uu = ( u*360-180) * 0.01745
    vv = ( v*180 -90) * 0.01745        
    
    # uv to 3D
    x =  np.cos(uu) * np.cos(vv)    
    y =  np.sin(uu) * np.cos(vv)
    z =  np.sin(vv)
    return x,y,z

def xyz_to_uv(x,y,z):
    theta   = np.arctan2(y,x)
    phi     = np.arcsin(z/(np.sqrt(x**2 +y**2+z**2)))

    theta   = (theta/ 0.01745 +180)/360
    phi     = (phi/0.01745 + 90)/180
    return theta,phi
    
def interplate_uv(u,v , count = 20):
    xs,ys,zs =[],[],[]

    for uu,vv in zip( u, v ):                    
        x,y,z = uv_to_xyz(uu,vv)
        xs.append(x)
        ys.append(y)
        zs.append(z)

   
    intp_x  = np.linspace(xs[0] , xs[1] , num=count )
    intp_y  = np.linspace(ys[0] , ys[1] , num=count )
    intp_z  = np.linspace(zs[0] , zs[1], num=count )

    # 3D to uv
    thetas  =[]
    phis    =[]
    for x,y,z in zip (intp_x, intp_y, intp_z):
        theta , phi = xyz_to_uv(x,y,z)
        thetas.append(theta)
        phis.append(phi)
    return thetas , phis

def rearng(x):    
    half_idx = len(x)//2
    x1= x[:half_idx]
    x2= x[half_idx:]
    if(isinstance(x , np.ndarray)):
        arr = np.zeros_like(x)
    else:
        arr = torch.zeros_like(x)
    arr[::2] = x1    
    arr[1::2] = x2      

    return arr
def rearrange_decoded(u,vt,vb):    
    us ,vts ,vbs= [],[],[]
    for batch_u , batch_vt , batch_vb in zip(u,vt,vb):   
        if(isinstance(batch_u , np.ndarray)):
            ru = rearng(batch_u)        
            rvt = rearng(batch_vt)
            rvb = rearng(batch_vb)
        else:
            ru = rearng(batch_u.detach().cpu().numpy())        
            rvt = rearng(batch_vt.detach().cpu().numpy())
            rvb = rearng(batch_vb.detach().cpu().numpy())

        us.append(ru)
        vts.append(rvt)
        vbs.append(rvb)
    
    return us,vts,vbs

'''
boxu = [np.array([0.9556, 0.9921])]
boxvt = [np.array([0.3810, 0.4109])]
boxvb = [np.array([0.7335, 0.6852])]
'''
#Cross Image Set
boxu = [np.array([0.7444, 1.0161])]
boxvt = [np.array([0.2220, 0.2190])]
boxvb = [np.array([0.8964, 0.8982])]
'''

# Multi door
boxu = [np.array([0.8111, 0.3778, 1.2295, 0.6528])]
boxvt = [np.array([0.4642, 0.2425, 0.4658, 0.2188])]
boxvb = [np.array([0.5735, 0.8770, 0.5703, 0.8927])]
'''

gt_boxu = [ np.array([0.7111, 0.3578, 1.1095, 0.6528])]
gt_boxvt =[ np.array([0.4642, 0.2425, 0.4658, 0.2188])]
gt_boxvb =[ np.array([0.5735, 0.8770, 0.5703, 0.8927])]

'''
boxu = [np.array([0.1222, 0.6778, 0.7111, 0.1364, 0.6922, 0.7348])]
boxvt = [np.array([0.4698, 0.4242, 0.4627, 0.4723, 0.4376, 0.4620])]
boxvb = [np.array([0.5649, 0.6636, 0.5813, 0.5593, 0.6373, 0.5829])]
'''
boxu , boxvt , boxvb = rearrange_decoded(boxu , boxvt, boxvb)
gt_boxu , gt_boxvt , gt_boxvb = rearrange_decoded(gt_boxu , gt_boxvt, gt_boxvb)
from shapely.validation import make_valid,explain_validity
from shapely.geometry import Polygon

#==============================================
# Note :     Shapely will works most of the time, but it sometimes getting invaild shape errors.
#            So we use pixel level IoU instead of Shapely.
#==============================================

def cal_poly_iou(poly_a , poly_b):
    
    if( len(poly_a) ==1 and len(poly_b) ==1): #比對的兩扇門都沒有跨畫面
        a_pg = Polygon(poly_a[0])
        b_pg = Polygon(poly_b[0])
        
        a_pg = a_pg.buffer(0)
        a_pg = a_pg.simplify(0.0001 ,preserve_topology=False)
        b_pg = b_pg.buffer(0)
        b_pg = b_pg.simplify(0.0001 , preserve_topology=False)

        poly_intersection   = a_pg.intersection(b_pg)
        poly_union          = a_pg.union(b_pg)
        if( poly_union.area== 0):
            iou=0
        else :
            iou                 = poly_intersection.area / poly_union.area        
        return iou
        pass

    else:
        iou_matrix = np.zeros((len(poly_a) , len(poly_b)))      
        for i , a_points in enumerate( poly_a):
            a_pg = Polygon(a_points)
            a_pg.buffer(0.0001)
            a_pg = a_pg.simplify(0.001 ,preserve_topology=False)

            if(not a_pg.is_valid):                
                a_pg = make_valid(a_pg)
                print("a",a_pg.is_valid , explain_validity(a_pg))

            for j , b_points in enumerate( poly_b):
                b_pg = Polygon(b_points)
                b_pg = b_pg.buffer(0)
                b_pg = b_pg.simplify(0.001 , preserve_topology=False)

                if(not b_pg.is_valid):
                    b_pg = make_valid(b_pg)
                    print("b" , b_pg.is_valid , explain_validity(b_pg))
                
                poly_intersection   = a_pg.intersection(b_pg)
                poly_union          = a_pg.union(b_pg)
                if( poly_union.area== 0):
                    iou =0
                else:                
                    iou                 = poly_intersection.area / poly_union.area
                iou_matrix[i][j] =  np.float32( iou)
                #print("iou    " ,iou    )
        total_iou = np.sum(iou_matrix)/2
        #print("iou_matrix" , iou_matrix)
        #print("total iou" , total_iou)

        return total_iou
    pass

def get_iou_matrix_distored(gt , pred):    
    iou_matrix = np.zeros((len(gt) , len(pred)))        
    for i , _gt in  enumerate(gt):        
        for j , _pred in enumerate(pred):            
            iou  = cal_poly_iou(_gt, _pred)
            iou_matrix[i][j] = np.float32( iou)
    
    return iou_matrix
    
