import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

import sys
import argparse
from model import *
sys.path.append('../')
from file_helper import *
import numpy as np


if __name__ =="__main__":
    parser = argparse.ArgumentParser(description="Run model")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--ep', type=int, default=50)
    parser.add_argument('--b', type=int, default=8)   # batch size
    parser.add_argument('--t', action='store_true' )  # test mode
    parser.add_argument('--db', type=str , default="ZillowImageFolder" )  # test mode
    args = parser.parse_args()

    train_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_train_visiable_all.json")
    test_file = os.path.join(os.path.dirname(os.getcwd()) , "anno" , "coco_test_visiable_all.json")

    train_loader , test_loader , train_dataset , test_dataset = get_loader(
        img_root_folder=args.db,
        try_load=True,
        train_json_path= train_file , 
        test_json_path= test_file,
        batch_size= args.b 
            )
    
    model = Detr(lr=1e-4, lr_backbone=1e-5, weight_decay=1e-4 ,train_loader= train_loader , test_loader= test_loader)    

    
    trainer = Trainer(devices=args.gpu, accelerator="gpu", max_epochs=args.ep, 
            gradient_clip_val=0.1, accumulate_grad_batches=8, log_every_n_steps=2 , fast_dev_run = args.t)

    trainer.fit(model)
    trainer.test(model)
