from torchvision import transforms
from PIL import Image
img_path=r"dataset\train\ants_image\6240338_93729615ec.jpg"
img_PIL=Image.open(img_path)
tensor_trans=transforms.ToTensor()
img_tensor=tensor_trans(img_PIL)