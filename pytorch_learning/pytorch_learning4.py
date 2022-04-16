from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
writer=SummaryWriter("logs")
image_path=r"dataset\train\bees_image\478701318_bbd5e557b8.jpg"
img_PIL=Image.open(image_path)
img_array=np.array(img_PIL)
writer.add_image("train",img_array,1,dataformats='HWC')
for i in range(100):
    writer.add_scalar("y=2x",2*i,i)
writer.close()