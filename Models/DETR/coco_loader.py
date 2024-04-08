import torch
import cv2
import torch
import supervision as sv
import os
abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)
import torchvision
from transformers import DetrForObjectDetection, DetrImageProcessor

CHECKPOINT = 'facebook/detr-resnet-50'
image_processor = DetrImageProcessor.from_pretrained(CHECKPOINT)

class CocoDetection(torchvision.datasets.CocoDetection):
    def __init__(
        self,
        image_directory_path: str,
        image_processor,
        train: bool = True,
        annotation_file_path=""
    ):
        #annotation_file_path = os.path.join(image_directory_path, annotation_file_path)
                
        print("annotation_file_path" , annotation_file_path , "image_directory_path" , image_directory_path)
        super(CocoDetection, self).__init__(image_directory_path, annotation_file_path)
        self.image_processor = image_processor

    def __getitem__(self, idx):
        images, annotations = super(CocoDetection, self).__getitem__(idx)
     
        image_id = self.ids[idx]
        annotations = {'image_id': image_id, 'annotations': annotations}
        encoding = self.image_processor(images=images, annotations=annotations, return_tensors="pt")
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target
    

from torch.utils.data import DataLoader
def collate_fn(batch):
    # DETR authors employ various image sizes during training, making it not possible
    # to directly batch together images. Hence they pad the images to the biggest
    # resolution in a given batch, and create a corresponding binary pixel_mask
    # which indicates which pixels are real/which are padding
    pixel_values = [item[0] for item in batch]
    #print("pixel_values" , pixel_values)
    encoding = image_processor.pad(pixel_values, return_tensors="pt")
    labels = [item[1] for item in batch]
    return {
        'pixel_values': encoding['pixel_values'],
        'pixel_mask': encoding['pixel_mask'],
        'labels': labels
    }



def get_loader ( img_root_folder  ,
                 train_json_path ,
                 test_json_path ,
                 batch_size = 4,
                 try_load = False ):
    
    TRAIN_DIRECTORY = img_root_folder
    #VAL_DIRECTORY = os.path.join(drive_path)
    TEST_DIRECTORY = img_root_folder
    
    TRAIN_DATASET = CocoDetection(
        image_directory_path=TRAIN_DIRECTORY,
        image_processor=image_processor,
        train=True,
        annotation_file_path= train_json_path )
    
    TEST_DATASET = CocoDetection(
        image_directory_path=TEST_DIRECTORY,
        image_processor=image_processor,
        train=False,
        annotation_file_path= test_json_path
        )

    print("Number of training examples:", len(TRAIN_DATASET))    
    print("Number of test examples:", len(TEST_DATASET))


    TRAIN_DATALOADER = DataLoader(dataset=TRAIN_DATASET, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    TEST_DATALOADER = DataLoader(dataset=TEST_DATASET, collate_fn=collate_fn, batch_size=batch_size, shuffle=False)
    
    # Test...    
    if(try_load):
        batch = next(iter(TRAIN_DATALOADER))
        batch = next(iter(TEST_DATALOADER))
    

    return TRAIN_DATALOADER , TEST_DATALOADER , TRAIN_DATASET , TEST_DATASET

