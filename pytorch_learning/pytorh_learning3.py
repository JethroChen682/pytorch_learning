from PIL import Image
import os
from torch.utils.data import Dataset
class MyData(Dataset):
    def __init__(self,root_dir,label_dir):
        self.root_dir=root_dir
        self.label_dir=label_dir+"_label"
        self.image_dir=label_dir+"_image"
        self.path=os.path.join(self.root_dir,self.image_dir)
        self.img_path=os.listdir(self.path)
        self.label_path_ori=os.path.join(self.root_dir,self.label_dir)
        self.label_path=os.listdir(self.label_path_ori)
    def __getitem__(self, idx):
        img_name=self.img_path[idx]
        label_name=img_name.split(".jpg")[0]+".txt"
        img_item_path=os.path.join(self.root_dir,self.image_dir,img_name)
        img_label_path=os.path.join(self.root_dir,self.label_dir,label_name)
        img=Image.open(img_item_path)
        f=open(img_label_path,"r")
        label=f.read()
        f.close()
        return img,label
    def __len__(self):
        return len(self.img_path)
root_dir=r"dataset\train"
ants_label_dir=r"ants"
bees_label_dir=r"bees"
ants_dataset=MyData(root_dir,ants_label_dir)
bees_dataset=MyData(root_dir,bees_label_dir)
traindata=ants_dataset+bees_dataset
