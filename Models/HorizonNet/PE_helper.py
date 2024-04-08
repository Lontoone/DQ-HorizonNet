import numpy as np
from sklearn import metrics
import cv2
import matplotlib.pyplot as plt
from horizon_utlis import *


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

'''
---------------------------------------------
|                                            |
|c---s                                 0------|
|     |                                 |     |
|     |                                 |     |
|     |                                 |     |
|----_0                                _n-----|
---------------------------------------------
* c = cross_idx
* s = segment_idx
* 0 = start idx
* _0 = start at btm idx,  = s + 1
    order : _0 -> _n
    _n is the last
* 
'''
def to_distorted_box(u,vt,vb , image = None , seg_count =30 , show_plt= True ,
                     return_mask=True  , h = 1024 , w=2048 ,return_img = False , fill_color = (255,0,0) , thickness = 3 , dotted = False):

    if return_mask or show_plt or return_img:            
        canvas = np.zeros((h,w,3)) if image is None else image

    for _u , _vt , _bv  in zip(u, vt , vb):
        cross_idx = -1        
        previous_x = 0        
        have_cross = _u[1]>1
        # Upper line
        all_points = [[None]*seg_count*2][0]
        thetas , phis= interplate_uv(_u,_vt , seg_count)
        i=0
        for t, p in zip(thetas , phis):         
            if (show_plt or return_img ) and dotted and i%5==0:   
                canvas = cv2.circle(canvas , (int(t*w) , int(p *h)) , 5 , fill_color ,-1 )            

            if(t< previous_x and have_cross):
                cross_idx = i 
            
            previous_x = t
            all_points[i] = [t , p]

            i+=1
        if (not have_cross):
            # Make sure it has the correct order
            sort_idx =  np.argsort(  np.asarray( all_points[:i])[:,0] )
            all_points[:i] =   np.asarray( all_points)[sort_idx]

        # Bottom line
        thetas , phis= interplate_uv(_u,_bv , seg_count)        
        i=1
        for t, p in zip(thetas , phis):
            if (show_plt or return_img ) and dotted and i%5==0:   
                canvas = cv2.circle(canvas , (int(t*w) , int(p *h)) , 5 , fill_color ,-1 )            
            all_points[len(all_points)- i] = [t , p]
            i+=1

        if (not have_cross):
            # Make sure it has the correct order
            sort_idx =  np.argsort(  np.asarray( all_points[i:])[:,0] )[::-1]
            all_points[i:] =   np.asarray( all_points )[sort_idx+i]
        
        # check cross      
        if(cross_idx >0):
            left_start_top = [0 ,all_points[cross_idx][1] ]
            left_start_btm = [0 ,all_points[seg_count+1][1] ]
            
            right_mid_top = [1 ,all_points[ cross_idx-1][1] ]            
            right_mid_btm = [1 ,all_points[-1][1] ]

            part_right = all_points[:cross_idx] +[right_mid_top] +[right_mid_btm]+ all_points[seg_count + (seg_count - cross_idx)  :] 
            
            part_left = [left_start_top] + \
                all_points[cross_idx: seg_count + (seg_count - cross_idx)] + [left_start_btm]

            
            part_left = np.array(part_left)
            part_right = np.array(part_right)

            right_min = part_right[0][0]

            right_clip_idx = np.where(part_right.flatten()[::2]<right_min)[0]

            if (right_clip_idx.size > 0):
                part_right[right_clip_idx,0]=1

            polys = [part_left.tolist() , part_right.tolist()]

        else:
            polys = [all_points]
        for poly in polys:            
            poly = np.array(poly)
            poly = poly.reshape((-1 , 2)) * np.tile(np.array([w , h]) , (poly.size//2 , 1) )
            poly = poly.astype('int32')    
            
            if return_mask :           
                canvas =  cv2.fillPoly(canvas, [poly], (255,255,255))                
            if return_img:
                canvas =  cv2.polylines(canvas, [poly], True, fill_color , thickness = thickness )
    if show_plt:
        if image is not None:
            plt.imshow(cv2.cvtColor( canvas , cv2.COLOR_BGR2RGB ))
        else:
            plt.imshow(canvas )
        plt.show() 
    if return_mask:
        return polys, canvas
    elif return_img:
        return polys,  canvas 
    else:
        return polys

def rearng(x):    
    half_idx = len(x)//2
    x1= x[:half_idx]
    x2= x[half_idx:]
    
    arr = np.zeros_like(x)
    arr[::2] = x1    
    arr[1::2] = x2      

    return arr
def rearrange_decoded(u,vt,vb):    
    us ,vts ,vbs= [],[],[]
    for batch_u , batch_vt , batch_vb in zip(u,vt,vb):                    
        ru = rearng(batch_u)        
        rvt = rearng(batch_vt)
        rvb = rearng(batch_vb)

        us.append(ru)
        vts.append(rvt)
        vbs.append(rvb)
    
    return us,vts,vbs

def get_iou_matrix_pixel_level(gts , preds ):
		iou_matrix = np.zeros((len(gts) , len(preds)))		

		for i , gt in enumerate(gts):						
			for j , pred in enumerate(preds):	
				intersect = np.count_nonzero(gt * pred)
				union = np.count_nonzero((gt + pred)>0 )
				if(union ==0):
					iou =0
				else:
					iou =  intersect / union

				iou_matrix[i][j] = iou
		return iou_matrix

def get_iou_matrix_polygon(gt , pred):    
    iou_matrix = np.zeros((len(gt) , len(pred)))        
    for i , _gt in  enumerate(gt):        
        for j , _pred in enumerate(pred):            
            iou  = cal_poly_iou(_gt, _pred)
            iou_matrix[i][j] = np.float32( iou)    
    return iou_matrix
    

class PR_Eval_Helper():    
    def __init__(self, writer=None , ep=0 , log_folder ="" , get_iou_fn = get_iou_matrix_pixel_level):
        self.iou_thresh = [0.05,0.5,0.75]
        self.gt_count=0
        self.all_iou=0
        self.results_per_batch = [{"tp":[],"fp":[]} for _ in self.iou_thresh]
        self.writer = writer
        self.ep = ep
        self.log_folder =log_folder
        self.get_iou_fn = get_iou_fn
        pass
  

    def eval_batch_pr( self, predict_result , gt ,  _debug_iteration =0 ):        

        self.predict_result = predict_result
        self.gt = gt
        
        self.gt_count+= len(gt)   
        best_iou = []
        
            
        iou_matrix = self.get_iou_fn(gt,predict_result)                
        pred_count = len(self.predict_result)
        
        for i,iou_thersh in enumerate( self.iou_thresh):           
            if pred_count >= 1:
                iou_thresh_mask = np.where(iou_matrix >= iou_thersh , 1 , 0 )                
                masked_iou_matrix = iou_matrix * iou_thresh_mask                
                each_gt_best_iou_idx = np.argmax(masked_iou_matrix , axis=1)                
                
                # filter multiple bboxs
                mono_gt = np.zeros_like(iou_matrix)
                mono_gt[[i for i in range(len(mono_gt))] , [each_gt_best_iou_idx]] = iou_matrix[[i for i in range(len(mono_gt))] , [each_gt_best_iou_idx]]
                
                mono_gt *= iou_thresh_mask  # set to 0 if under threshold
                # log miou
                if(i==0):
                    self.all_iou += np.sum(mono_gt)


                # find best iou for each GT
                # Find TP , FN
                pred_filtered_mask = np.zeros(pred_count)                
                for row_gt in mono_gt:
                    _max_idx = np.argmax(row_gt)
                    if( row_gt[_max_idx]>0 ):
                        pred_filtered_mask[_max_idx]=1          

                    if(i==0):
                        best_iou.append(row_gt[_max_idx].astype(float))
           
                tp_list = np.where(pred_filtered_mask > 0 , 1 , 0 )   
                fp_list = np.where(pred_filtered_mask ==0 , 1 , 0 )    
                
                self.results_per_batch[i]['tp'].append(tp_list.flatten())
                self.results_per_batch[i]['fp'].append(fp_list.flatten())
                
            else:
                batch_ap = 0
        return best_iou
          

    
    def list_to_pr_auc(self, tp_list , fp_list , gt_count):
        tp = np.cumsum(tp_list)     
        fp = np.cumsum(fp_list) 
        
        all_prediction = tp+fp
        precision = tp / all_prediction
        recall = tp / gt_count            
        self.all_prediction = precision
        
        recall = np.insert(recall , 0 , 0)
        precision = np.insert(precision , 0 , 1)


        auc = metrics.auc(recall,precision)
        return precision , recall,auc
    
    def get_all_pr(self  , logger, show_plt = True):
        # combine each batch result and sort by scores
        self.final_result_dict =  [{} for _ in self.iou_thresh]        
        for i , thresh in enumerate(self.iou_thresh):
            if self.gt_count > 0 :            
                all_tp = np.concatenate(self.results_per_batch[i]['tp'][:])
                all_fp = np.concatenate(self.results_per_batch[i]['fp'][:])

                sum_tp = np.sum(all_tp)
                sum_fp = np.sum(all_fp)

                recall_rate = sum_tp / self.gt_count
                precision_rate = sum_tp / (sum_tp + sum_fp)
                                  
                precision , recall , auc = self.list_to_pr_auc(all_tp , all_fp , self.gt_count)
            else:
                precision = []
                recall=[]
                auc=0
                recall_rate =0
                precision_rate=0

            self.final_result_dict[i]=(
                {"iou_thresh": thresh ,
                    "recall":recall,
                    "precision":precision,
                    "recall_rate": recall_rate,
                    "precision_rate": precision_rate,
                    "ap":auc} 
            )
            print(f"ap_{thresh}",auc)
        if self.writer or logger is not None:
            self.write_tensorboard(logger ,show_plt= show_plt)
        
        mIou =  self.all_iou/self.gt_count
        #return precision,recall,auc
        return mIou
    
    def write_tensorboard(self, logger, subName ="sub"  , show_plt = True ):        
        writer = self.writer
        for i , thresh in enumerate(self.iou_thresh):                     
            prcs = self.final_result_dict[i]["precision"] 
            recs = self.final_result_dict[i]["recall"] 
            
            step = 0
            for prc, rec in zip(prcs , recs ):
                if(self.writer is not None):
                    
                    logger.log  (f"Eval/Precision_{thresh}",{f"ep{self.ep}":prc} )
                    logger.log  (f"Eval/Recall_{thresh}" ,{f"ep{self.ep}":rec})
                    logger.log  (f"Eval/AUC_{thresh}" ,{f"ep{self.ep}": prc})  # tensor board bug                                                    
                
                step+=1
            logger.log(f"Eval/Precision_rate_{thresh}", self.final_result_dict[i]['precision_rate'])
            logger.log(f"Eval/Recall_rate_{thresh}" ,   self.final_result_dict[i]['recall_rate'] )
            logger.log(f"Eval/AUC_{thresh}" ,  self.final_result_dict[i]['ap']  )  # tensor board bug                                                    
            logger.log(f"Eval/mIou" ,  self.all_iou/self.gt_count  )  # tensor board bug                                                    

            print("all_iou" , self.all_iou)
            print("mIOU" , self.all_iou/self.gt_count)
            
            if(show_plt):
                plt.plot( recs ,prcs )   
                plt.title(f"PR_curve-{thresh} ap : {self.final_result_dict[i]['ap']}")
                plt.savefig(self.log_folder+f"/_PR_curve-{thresh}-{subName}-ep{self.ep}.jpg")
                plt.show()
            
            
        #self.writer.close()
        pass


if __name__ =="__main__":
    #////////////// TEST VALUE ///////////////

    # Multi door
    boxu = [np.array([0.2111, 0.3778, 0.2295, 0.6528])]
    boxvt = [np.array([0.4642, 0.2425, 0.4658, 0.2188])]
    boxvb = [np.array([0.5735, 0.8770, 0.5703, 0.8927])]
    '''
    # test set 2
    boxu = [np.array([0.9556, 0.9921])]
    boxvt = [np.array([0.3810, 0.4109])]
    boxvb = [np.array([0.7335, 0.6852])]
    '''

    '''
    # test set 3
    boxu = [np.array([0.1222, 0.6778, 0.7111, 0.1364, 0.6922, 0.7348])]
    boxvt = [np.array([0.4698, 0.4242, 0.4627, 0.4723, 0.4376, 0.4620])]
    boxvb = [np.array([0.5649, 0.6636, 0.5813, 0.5593, 0.6373, 0.5829])]
    '''

    boxu , boxvt , boxvb = rearrange_decoded(boxu , boxvt, boxvb)

    for batched_uvv in zip(boxu , boxvt , boxvb ):
        print("batched_uvv" , batched_uvv)
        u = batched_uvv[0].reshape(-1,2)
        vt = batched_uvv[1].reshape(-1,2)
        vb = batched_uvv[2].reshape(-1,2)    
        to_distorted_box(u , vt , vb )