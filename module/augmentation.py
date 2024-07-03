import torch
from torch import Tensor
import math
import numbers
import warnings
import numpy as np
from typing import List, Tuple
from collections.abc import Sequence
from PIL import ImageFilter, ImageOps
import random
import numpy as np
import torchvision.transforms as transforms

from torchvision.transforms import functional as F, InterpolationMode
from torchvision.transforms.functional import _interpolation_modes_from_int

class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class ComposeWithBox:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        img, bbox = self.transforms[0](img)
        img, isFlip = self.transforms[1](img)

        for t in self.transforms[2:]:
            img = t(img)

        return img, bbox, isFlip

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class ComposeFromBox:

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):

        img, box2, box_overlap, overlapArea = self.transforms[0](img)
        img, isFlip = self.transforms[1](img)

        for t in self.transforms[2:]:
            img = t(img)

        return img, box2, box_overlap, overlapArea, isFlip

    def __repr__(self) -> str:
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

class TwoCropsTransformBox:
    """Take two random crops of one image"""

    def __init__(self, patch_size = 16, divide_size = 32, divide_size2 = 112, image_size = 224, max_size = 10, max_size2 = 10, max_size3 = 2):
        self.patch_size = patch_size
        self.divide_size = divide_size
        self.divide_size2 = divide_size2
        self.image_size = image_size
        self.max_size = max_size
        self.max_size2 = max_size2
        self.max_size3 = max_size3

    def __call__(self, x):
        base_transform1 =  ComposeWithBox([
                                RandomResizedCropBox(self.image_size, scale=(0.2, 1.)),
                                RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                # transforms.RandomSolarize(threshold=128, p=0.2),
                                transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                    ])

        im1, box1, isFlip_box1 = base_transform1(x)
        base_transform2  = ComposeFromBox([
                                RandomResizedCropFromBox(self.image_size, box1 ,scale=(0.2, 1.)),
                                RandomHorizontalFlip(),
                                transforms.RandomApply([
                                    transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)
                                ], p=0.8),
                                transforms.RandomGrayscale(p=0.2),
                                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
                                transforms.RandomSolarize(threshold=128, p=0.2),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                                # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                                            ])
        im2, box2, box_overlap, overlapArea, isFlip_box2 = base_transform2(x)
        box1 = torch.tensor(box1)
        box2 = torch.tensor(box2)
        patch_indexs1, patch_indexs2, ious = self.get_iou_from_box(box1, box2, box_overlap, self.patch_size, self.image_size, self.max_size, isFlip_box1, isFlip_box2, 12, 8, (2, 8), (2, 6))
        patch_indexs3, patch_indexs4, ious2 = self.get_iou_from_box(box1, box2, box_overlap, self.divide_size, self.image_size, self.max_size2, isFlip_box1, isFlip_box2, 8, 4, (2, 4), (1, 3))
        patch_indexs5, patch_indexs6, ious3 = self.get_iou_from_box(box1, box2, box_overlap, self.divide_size2, self.image_size, self.max_size3, isFlip_box1, isFlip_box2, 1, 1, (1, 1), (1, 1))
        return [im1, im2, patch_indexs1, patch_indexs2, ious, patch_indexs3, patch_indexs4, ious2, patch_indexs5, patch_indexs6, ious3]


    def get_iou_from_box(self, box1, box2, box_overlap, patch_size, image_size, max_size, isFlip_box1 = False, isFlip_box2 = False, step_ratio1=8, step_ratio2=4, minmax1=(1, 16), minmax2=(1, 16)):

        # convert the overlapping box into one relative to box2
        ir1, jr1, hr1, wr1 = self.get_relative_box(box1, box_overlap, image_size)
        
        length = image_size // patch_size
        patch_num = length ** 2
        _, _, box1_h, box1_w = box1
        _, _, box2_h, box2_w = box2
        _, _, box_overlap_h, box_overlap_w = box_overlap
        iou = (box_overlap_h * box_overlap_w) / (box1_h * box1_w + box2_h * box2_w - box_overlap_h * box_overlap_w )
        step1 = min(max(int(iou*step_ratio1), minmax1[0]), minmax1[1])
        step2 = min(max(int(iou*step_ratio2), minmax2[0]), minmax2[1])
        s_box1 =  box1_h * box1_w / patch_num
        s_box2 =  box2_h * box2_w / patch_num
        size_length = 0       
        patch_indexs1 = []
        patch_indexs2 = []
        ious = []
        
        # obtain the overlapped box index of begin and end
        patch_begin_j = int(ir1 // patch_size) 
        patch_begin_i = int(jr1 // patch_size)

        if (ir1+wr1) % patch_size == 0 or (ir1+wr1) > image_size: 
            patch_end_j = int((ir1+wr1) // patch_size) - 1
        else:         
            patch_end_j = int((ir1+wr1) // patch_size) 

        if (jr1+hr1) % patch_size == 0 or (jr1+hr1) > image_size:
            patch_end_i = int((jr1+hr1) // patch_size) - 1
        else:
            patch_end_i = int((jr1+hr1) // patch_size)  


        for i in range(patch_begin_i, patch_end_i + 1, step1):
            for j in range(patch_begin_j, patch_end_j + 1, step2):
                # compute the index of box2 overlapped with the current box1
                patch_index = i * length + length - j -1 \
                    if isFlip_box1 else i * length + j
                
                # the overlapping area of the current box with patch box
                patch_box, overArea = isOverlap((ir1, jr1, hr1, wr1), (j*patch_size, i*patch_size, patch_size, patch_size))

                # convert overlapping area into absoluate one, then to relative one with respective to box2
                patch_box = self.get_absoluate_box(box1, patch_box, image_size)
                ir2, jr2, hr2, wr2 = self.get_relative_box(box2, patch_box, image_size)
                
                # check whether out of border
                if(jr2<0):
                    jr2=0
                if(ir2<0):
                    ir2=0
                patch_begin_i2 = int(jr2 // patch_size)
                patch_begin_j2 = int(ir2 // patch_size) 

                if (ir2+wr2) % patch_size == 0 or (ir2+wr2) > image_size:
                    patch_end_j2 = int((ir2+wr2) // patch_size) - 1 
                else:
                    patch_end_j2 = int((ir2+wr2) // patch_size)       
                if (jr2+hr2) % patch_size == 0 or (jr2+hr2) > image_size :
                    patch_end_i2 = int((jr2+hr2) // patch_size) - 1
                else:
                    patch_end_i2 = int((jr2+hr2) // patch_size) 

                # total_ratio = 0
                # max_ratio = 0
                for p in range(patch_begin_i2, patch_end_i2 + 1):
                    # flag = 0
                    for q in range(patch_begin_j2, patch_end_j2 + 1):
                        patch_index2 = p * length + length - q - 1 \
                            if isFlip_box2 else p * length + q                        
                        _, overArea1 = isOverlap((q * patch_size, p*patch_size, patch_size, patch_size), (ir2, jr2, hr2, wr2))   
                        value = overArea * overArea1 / (s_box1 + s_box2)
                        # iou = value * s_box1 / (s_box1 + s_box2 - value * s_box1)
                        iou = value * s_box1
                        if (iou>0):
                            size_length = size_length + 1
                            patch_indexs1.append(patch_index)
                            patch_indexs2.append(patch_index2)
                            ious.append(iou)

        patch_indexs1, patch_indexs2 = np.array(patch_indexs1), np.array(patch_indexs2)
        ious = np.array(ious)
        if size_length > max_size :
            idxs_sorted= np.argsort(ious) # ascending sort
            idxs_sorted = idxs_sorted[-max_size:]
            patch_indexs1, patch_indexs2 = patch_indexs1[idxs_sorted], patch_indexs2[idxs_sorted]
            ious = ious[idxs_sorted]
            size_length = max_size
        else:
            patch_indexs1 = np.concatenate((patch_indexs1, np.zeros((max_size - size_length))), axis=0)
            patch_indexs2 = np.concatenate((patch_indexs2, np.zeros((max_size - size_length))), axis=0)
            ious = np.concatenate((ious, np.zeros((max_size - size_length))), axis=0)

        return torch.tensor(patch_indexs1,dtype=torch.long), torch.tensor(patch_indexs2,dtype=torch.long), torch.tensor(ious)


    def get_iou_from_box_random(self, box1, box2, box_overlap, patch_size, image_size, max_size, isFlip_box1 = False, isFlip_box2 = False):

        # convert the overlapping box into one relative to box2
        ir1, jr1, hr1, wr1 = self.get_relative_box(box1, box_overlap, image_size)
        
        length = image_size // patch_size
        patch_num = length ** 2
        _, _, box1_h, box1_w = box1
        _, _, box2_h, box2_w = box2
        s_box1 =  box1_h * box1_w / patch_num
        s_box2 =  box2_h * box2_w / patch_num
   
        patch_indexs1 = []
        patch_indexs2 = []
        ious = []

        # obtain the overlapped box index of begin and end
        patch_begin_j = int(ir1 // patch_size) 
        patch_begin_i = int(jr1 // patch_size)

        if (ir1+wr1) % patch_size == 0 or (ir1+wr1) >= image_size: 
            patch_end_j = int((ir1+wr1) // patch_size) - 1
        else:         
            patch_end_j = int((ir1+wr1) // patch_size) 

        if (jr1+hr1) % patch_size == 0 or (jr1+hr1) >= image_size:
            patch_end_i = int((jr1+hr1) // patch_size) - 1
        else:
            patch_end_i = int((jr1+hr1) // patch_size)  

        rand_j = np.random.randint(patch_begin_j, patch_end_j + 1, size=max_size, dtype=int)
        rand_i = np.random.randint(patch_begin_i, patch_end_i + 1, size=max_size, dtype=int)


        for index in range(0, max_size):
            # compute the index of box2 overlapped with the current box1
            patch_index = rand_i[index] * length + length - rand_j[index] -1 \
                if isFlip_box1 else rand_i[index] * length + rand_j[index]
                
            # the overlapping area of the current box with patch box
            patch_box, overArea = isOverlap((ir1, jr1, hr1, wr1), (rand_j[index]*patch_size, rand_i[index]*patch_size, patch_size, patch_size))

             # convert overlapping area into absoluate one, then to relative one with respective to box2
            patch_box = self.get_absoluate_box(box1, patch_box, image_size)
            ir2, jr2, hr2, wr2 = self.get_relative_box(box2, patch_box, image_size)

            if(jr2<0):
                jr2=0
            if(ir2<0):
                ir2=0
            patch_begin_i2 = int(jr2 // patch_size)
            patch_begin_j2 = int(ir2 // patch_size) 
            if patch_begin_i2 >= length:
                patch_begin_i2 = length -1 
            if patch_begin_j2 >= length:
                patch_begin_j2 = length -1                

            if  (ir2+wr2) % patch_size == 0 or (ir2+wr2) >= image_size:
                patch_end_j2 = int((ir2+wr2) // patch_size) - 1 
            else:
                patch_end_j2 = int((ir2+wr2) // patch_size)       
            if  (jr2+hr2) % patch_size == 0 or (jr2+hr2) >= image_size :
                patch_end_i2 = int((jr2+hr2) // patch_size) - 1
            else:
                patch_end_i2 = int((jr2+hr2) // patch_size) 

            if patch_end_i2 < 0 or patch_end_j2 < 0:
                print('e2')
                print(ir1, jr1, hr1, wr1)
                print(box1)
                print(box2)
                print(box_overlap)

            if patch_begin_i2 < 0 or patch_begin_j2 < 0:
                print('b2')
                print(ir1, jr1, hr1, wr1)
                print(box1)
                print(box2)
                print(box_overlap)

            p = np.random.randint(patch_begin_i2, patch_end_i2 + 1, dtype=int)
            q = np.random.randint(patch_begin_j2, patch_end_j2 + 1, dtype=int)
            patch_index2 = p * length + length - q - 1 \
                if isFlip_box2 else p * length + q                        
            _, overArea1 = isOverlap((q * patch_size, p*patch_size, patch_size, patch_size), (ir2, jr2, hr2, wr2))   
            value = overArea * overArea1
            iou = value * s_box1 / (s_box1 + s_box2 - value * s_box1)
            patch_indexs1.append(patch_index)
            patch_indexs2.append(patch_index2)
            ious.append(iou)
            if patch_index >= patch_num or patch_index2 >= patch_num or patch_index < 0 or patch_index2 <0:
                print(patch_index, patch_index2 )
                print(patch_begin_i2, patch_end_i2, patch_begin_j2, patch_end_j2)
                print(patch_begin_i, patch_end_i, patch_begin_j, patch_end_j)
        patch_indexs1, patch_indexs2 = np.array(patch_indexs1), np.array(patch_indexs2)
        ious = np.array(ious)       
        return torch.tensor(patch_indexs1,dtype=torch.long), torch.tensor(patch_indexs2,dtype=torch.long), torch.tensor(ious)



    
    def get_relative_box(self, box, box_overlap, image_size):
        # both box and box_overlap are absoluate box on the original image, box_overlap is converted into relative box based on box
        i1, j1, h1, w1 = box
        i2, j2, h2, w2 = box_overlap
        i3 = (i2 - i1)/w1 * image_size
        j3 = (j2 - j1)/h1 * image_size
        h3 = h2/h1 * image_size
        w3 = w2/w1 * image_size
        return i3, j3, h3, w3 

    def get_absoluate_box(self, box, box_overlap, image_size):
        # box_overlap are relative box based on box, box is absoluate box on the original image, box_overlap is converted into absoluate box
        i1, j1, h1, w1 = box
        i2, j2, h2, w2 = box_overlap
        i3 = i2/image_size * w1 + i1
        j3 = j2/image_size * h1 + j1
        h3 = h2/image_size * h1
        w3 = w2/image_size * w1

        return i3, j3, h3, w3


def _setup_size(size, error_msg):
    if isinstance(size, numbers.Number):
        return int(size), int(size)

    if isinstance(size, Sequence) and len(size) == 1:
        return size[0], size[0]

    if len(size) != 2:
        raise ValueError(error_msg)

    return size


class RandomResizedCropBox(torch.nn.Module):

    def __init__(self, size, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float]) -> Tuple[int, int, int, int]:
        width, height = F.get_image_size(img)
        # width, height = F._get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w

    def forward(self, img):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, (j, i, h, w)

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string



class RandomResizedCropFromBox(torch.nn.Module):

    def __init__(self, size, box1, scale=(0.08, 1.0), ratio=(3.0 / 4.0, 4.0 / 3.0), interpolation=InterpolationMode.BILINEAR):
        super().__init__()
        self.size = _setup_size(size, error_msg="Please provide only two dimensions (h, w) for size.")

        if not isinstance(scale, Sequence):
            raise TypeError("Scale should be a sequence")
        if not isinstance(ratio, Sequence):
            raise TypeError("Ratio should be a sequence")
        if (scale[0] > scale[1]) or (ratio[0] > ratio[1]):
            warnings.warn("Scale and ratio should be of kind (min, max)")

        # Backward compatibility with integer value
        if isinstance(interpolation, int):
            warnings.warn(
                "Argument interpolation should be of type InterpolationMode instead of int. "
                "Please, use InterpolationMode enum."
            )
            interpolation = _interpolation_modes_from_int(interpolation)

        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio
        self.box1 = box1

    @staticmethod
    def get_params(img: Tensor, scale: List[float], ratio: List[float], box1) -> Tuple[int, int, int, int]:
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        
        
        
        count = 0
        while(True):
            count = count + 1
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()
            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))       
            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                box2 = (j, i, h ,w)
                box_overlap, overlapArea = isOverlap(box2, box1)
                # if(overlapArea>0 and overlapArea< 0.5):
                if(overlapArea > 0):
                    _, _, box1_h, box1_w = box1
                    _, _, box2_h, box2_w = box2
                    _, _, box_overlap_h, box_overlap_w = box_overlap
                    iou = (box_overlap_h * box_overlap_w) / (box1_h * box1_w + box2_h * box2_w - box_overlap_h * box_overlap_w )
                    if (iou > 0.2):
                        return box2, box_overlap, overlapArea
            elif (count > 1000):
                box2 = (0, 0, height, width)
                box_overlap, overlapArea = isOverlap(box2, box1)
                return box2, box_overlap, overlapArea

    def forward(self, img):
        box2, box_overlap, overlapArea = self.get_params(img, self.scale, self.ratio, self.box1)
        j, i, h, w = box2
        img = F.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        return img, box2, box_overlap, overlapArea 

    def __repr__(self) -> str:
        interpolate_str = self.interpolation.value
        format_string = self.__class__.__name__ + f"(size={self.size}"
        format_string += f", scale={tuple(round(s, 4) for s in self.scale)}"
        format_string += f", ratio={tuple(round(r, 4) for r in self.ratio)}"
        format_string += f", interpolation={interpolate_str})"
        return format_string



def isOverlap(box1, box2, return_box = False):
    i1, j1, h1, w1 =box1
    i2, j2, h2, w2 =box2 
    if ((i1 + w1  > i2) and (i2 + w2 > i1) and (j1 + h1 > j2) and (j2 + h2 > j1)):
        i3 = max(i1, i2)
        j3 = max(j1, j2)
        w3 = min((i1+w1), (i2+w2)) - i3
        h3 = min((j1+h1), (j2+h2)) - j3
        if return_box:
            return (i3, j3, h3, w3)
        return (i3, j3, h3, w3), (h3*w3)/(h2*w2)
    else:
        if return_box:
            return (0, 0, 0, 0)
        return (0, 0, 0, 0), 0

class RandomHorizontalFlip(torch.nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
    def forward(self, img):
        if torch.rand(1) < self.p:
            return F.hflip(img), True
        return img, False

eval_transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

eval_transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
