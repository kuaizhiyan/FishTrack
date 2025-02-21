from typing import Any
from torchvision.transforms import *

from PIL import Image
import random
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
import random
from mmcv.transforms import BaseTransform
from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class MMGEA(BaseTransform):
    def __init__(self,probability=0.7,sl=0.2,sh=0.7,rl=1,rh=3,thetal=-89,thetah=89,mean=[0.4914, 0.4822, 0.4465],density=0.5,scalar=10) -> None:
        '''     
        Implementation of GEA
        Param:
            probability: Probability of triggering GEA
            sl,sh: Legal erasing ratio range
            rl,rh: Variance ratio range
            thetal,thetah: Rotation range
            mean: [] value of replace pixel, default = [0.4914, 0.4822, 0.4465] 
            density: Density of Gaussian sampling
            scalar: Mapping scale of sampling
        '''
        self.probability = probability
        self.sl = sl
        self.sh = sh
        self.rl = rl
        self.rh = rh
        self.thetal = thetal
        self.thetah = thetah
        self.mean = mean
        self.density = density
        self.scalar = scalar
    
    def transform(self, results: dict) -> Any:
        if random.uniform(0,1) > self.probability:
            return results

        for _ in range(100):
            area = results['img'].shape[0] * results['img'].shape[1]
            
            # generate parameters
            S_e = random.uniform(self.sl,self.sh) * area
            r_e = random.uniform(self.rl,self.rh)
            theta = random.uniform(self.thetal,self.thetah) * np.pi / 180   

            # generate gaussian kernel        
            w = np.sqrt(S_e / r_e) / 5         
            h = np.sqrt(S_e * r_e) / 5         
            sigma1 = 1              
            sigma2 = r_e * sigma1
            
            # generate legal erasing area
            phi = np.arctan(1 / r_e)
            hypotenuse = np.sqrt((5*w)**2+(5*h)**2)   
            if theta >= 0:
                W = hypotenuse * np.sin(theta+phi)
                H = hypotenuse * np.cos(theta-phi)
            else :
                _theta = -1 * theta
                W = hypotenuse * np.sin(_theta+phi)
                H = hypotenuse * np.cos(_theta-phi)
            W = int(W)
            H = int(H)
            
            # sampling with gaussian kernel 
            mask = self.generate_gaussian_mask(sigma1,sigma2,theta,W,H,self.density,self.scalar)
                                 
            # pixel replacement
            if W // 2 < results['img'].shape[1] and H // 2 < results['img'].shape[0]:
                # generate expand image
                expand_img = np.zeros((results['img'].shape[0]+H,results['img'].shape[1]+W,results['img'].shape[2]))
                expand_img[H//2:H//2+results['img'].shape[0],W//2:W//2+results['img'].shape[1],:] = results['img'][:,:,:]
                x1 = random.randint(0, expand_img.shape[0] - H)
                y1 = random.randint(0, expand_img.shape[1] - W)
                
                # erase pixel by mask
                if expand_img.shape[2] == 3:          
                    mask = np.expand_dims(mask,axis=2)
                    expand_img[ x1:x1+H, y1:y1+W,:] = expand_img[x1:x1+H, y1:y1+W,:] * (1-mask)
                else:
                    expand_img[x1:x1+H, y1:y1+W,:] = expand_img[x1:x1+H, y1:y1+W,:] * (1-mask)
                    add_mask = np.ones(mask.size())
                    add_mask = add_mask * mask
                    add_mask[:,:,0] = add_mask[:,:,0] * self.mean[0]
                    expand_img[x1:x1+H, y1:y1+W,:] = expand_img[x1:x1+H, y1:y1+W,:] + add_mask
                    
                results['img'][:,:,:] = expand_img[H//2:H//2+results['img'].shape[0],W//2:W//2+results['img'].shape[1],:]
                return results
        return results

    def generate_gaussian_mask(self,sigma1,sigma2,theta,w,h,density=0.5,scalar=10):
        '''
        Implementation of Gaussian sampling
        Param:
            sigma1: variance of coordinate x
            sigma2: variance of coordinate y
            theta:  rotation angle
            w: width of legal erasing area
            h: height of legal erasing area
            density : sampling density
            scalar: scalar of coordinate mapping
        retrun:
            sampling mask
        '''
        w = int(w)
        h = int(h)
        
        if np.abs(theta) > 3.14:
            theta = theta * np.pi / 180
        
        # scalar matrix
        scalarMatrix=np.dot(np.matrix([[sigma1**2,0],[0,sigma2**2]]),np.identity(2))

        # rotation matrix
        rotationMatrix=np.matrix([[np.cos(theta),-1*np.sin(theta)],
                                [np.sin(theta),np.cos(theta)]])
        
        # covariance matrix
        covMatrix=np.dot(np.dot(rotationMatrix,scalarMatrix),rotationMatrix.transpose()) 
        
        # sample
        pts = np.random.multivariate_normal([0, 0], covMatrix, size=int(w*h*density))
        X = torch.Tensor(pts[:,0])      
        Y = torch.Tensor(pts[:,1])      
        locs = torch.stack((Y,X),dim=1)
        
        # mapping
        pts = (locs * scalar).int()            
        pts[:,0] = pts[:,0] + h //2     # h
        pts[:,1] = pts[:,1] + w //2     # w
        
        # filter illegal points
        select_mask = (pts[:,0]>=0)&(pts[:,0]<h)&(pts[:,1]>=0)&(pts[:,1]<w)
        pts = pts[select_mask]              
        pts = torch.unique(pts,dim=0)       
        
        # make the mask
        mask = torch.zeros(h,w)   
        lx = torch.LongTensor(pts[:,0].numpy()) 
        ly = torch.LongTensor(pts[:,1].numpy()) 
        replace_value = torch.ones_like(pts[:,0],dtype=mask.dtype)
        mask = mask.index_put((lx,ly),replace_value)
        mask = mask.numpy()
        return mask

@TRANSFORMS.register_module()
class GridMask(BaseTransform):
    """
    GridMask 数据增强，用于遮挡图像部分区域，提高模型的鲁棒性。
    该方法会在 `results['img']` 上应用网格遮挡，并返回修改后的 `results`。
    """
    def __init__(self, use_h=True, use_w=True, d1=24, d2=56, ratio=0.5, rotate=0, offset=False):
        """
        初始化 GridMask 参数。
        
        Args:
            use_h (bool): 是否在垂直方向应用 GridMask。
            use_w (bool): 是否在水平方向应用 GridMask。
            d1 (int): 最小网格间距。
            d2 (int): 最大网格间距。
            ratio (float): 遮挡区域的比例。
            rotate (int): 旋转角度。
            offset (bool): 是否随机偏移网格。
        """
        self.use_h = use_h
        self.use_w = use_w
        self.d1 = d1
        self.d2 = d2
        self.ratio = ratio
        self.rotate = rotate
        self.offset = offset

    def transform(self, results: dict) -> dict:
        """
        在图像上应用 GridMask。
        
        Args:
            results (dict): MMDetection 数据增强的输入字典。
        
        Returns:
            dict: 处理后的 `results` 字典。
        """
        img = results['img']
        img_h, img_w = img.shape[1],img.shape[0]
        
        d = random.randint(self.d1, self.d2)  # 选择网格大小
        l = int(d * self.ratio)  # 计算遮挡区域的大小

        mask = np.ones((img_h, img_w), dtype=np.float32)
        
        if self.offset:
            offset_x = random.randint(0, d)
            offset_y = random.randint(0, d)
        else:
            offset_x, offset_y = 0, 0

        for y in range(offset_y, img_h, d):
            for x in range(offset_x, img_w, d):
                y1, y2 = y, min(y + l, img_h)
                x1, x2 = x, min(x + l, img_w)
                mask[y1:y2, x1:x2] = 0  # 应用遮挡

        if self.rotate > 0:
            center = (img_w // 2, img_h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, self.rotate, 1.0)
            mask = cv2.warpAffine(mask, rotation_matrix, (img_w, img_h))

        mask = mask[:, :, np.newaxis]  # 添加通道维度
        results['img'] = img * mask  # 应用 GridMask
        
        return results

@TRANSFORMS.register_module()
class HideAndSeek(BaseTransform):
    """
    Hide-and-Seek 数据增强，随机遮挡图像的一部分，提高模型的鲁棒性。
    该方法会在 `results['img']` 上应用随机区域遮挡，并返回修改后的 `results`。
    """
    def __init__(self, grid_size=8, hide_prob=0.5):
        """
        初始化 Hide-and-Seek 参数。
        
        Args:
            grid_size (int): 图像划分的网格数量（每行每列的划分数）。
            hide_prob (float): 每个网格被遮挡的概率。
        """
        self.grid_size = grid_size
        self.hide_prob = hide_prob

    def transform(self, results: dict) -> dict:
        """
        在图像上应用 Hide-and-Seek。
        
        Args:
            results (dict): MMDetection 数据增强的输入字典。
        
        Returns:
            dict: 处理后的 `results` 字典。
        """
        img = results['img']
        img_w, img_h = img.shape[:2]
        
        grid_h = img_h // self.grid_size
        grid_w = img_w // self.grid_size
        
        mask = np.ones((img_h, img_w), dtype=np.float32)
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                if random.random() < self.hide_prob:
                    y1, y2 = i * grid_h, (i + 1) * grid_h
                    x1, x2 = j * grid_w, (j + 1) * grid_w
                    mask[y1:y2, x1:x2] = 0  # 应用遮挡
        
        mask = mask[:, :, np.newaxis]  # 添加通道维度
        results['img'] = img * mask  # 应用 Hide-and-Seek
        
        return results