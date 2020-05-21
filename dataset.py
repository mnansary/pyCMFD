"""
@author: MD.Nazmuddoha Ansary
"""
from __future__ import print_function
from termcolor import colored


import os
import numpy as np 
from glob import glob
import random
import cv2
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt 


class MICC_F2000(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='MICC_F2000'):
        self.data_dir=data_dir

        self.save_dir=os.path.join(save_dir,'DataSets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,iden)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        

        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'targets')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)


        self.tmpl_dir=os.path.join(self.save_dir,'Templates')
        if not os.path.exists(self.tmpl_dir):
            os.mkdir(self.tmpl_dir)
        
        self.image_dim=image_dim
        # Dataset Specific
        self.tamper_iden='tamp'
        self.orig_iden='_scale'
        self.base_tamp_iden='tamp1.jpg'
        self.prb_idens=['P1000231','DSCN47']
        self.data_count=0
        print(colored('Initializing:{}'.format(iden),'green'))

    def __renameProblematicFile(self):
        file_to_rename='nikon7_scale.jpg'
        proper_name='nikon_7_scale.jpg'
        try:
            if os.path.exists(os.path.join(self.data_dir,file_to_rename)):
                os.rename(os.path.join(self.data_dir,file_to_rename),os.path.join(self.data_dir,proper_name))
               
        except Exception as e:
            print(colored('!!! An exception occurred while renaming {}'.format(file_to_rename),'red'))
            print(colored(e,'green'))
        

    def __listFiles(self):
        self.tampered_files=[]
        self.tamper_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            if base_name not in self.prb_idens:
                self.tamper_idens.append(base_name)
                self.tampered_files.append(file_name)
            
        self.original_files=[]
        self.orig_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            if base_name not in self.prb_idens:
                self.orig_idens.append(base_name)
                self.original_files.append(file_name)
        


    def __create_ds(self):
        for iden in tqdm(self.orig_idens):
            #<-Saves the templates
            if iden in self.tamper_idens:
                base_image=self.original_files[int(self.orig_idens.index(iden))]
                img_data=cv2.imread(base_image,0)
                idx=[i for i, e in enumerate(self.tamper_idens) if e == iden] 
                for id_s in idx:
                    tmp_file=self.tampered_files[id_s]
                    if self.base_tamp_iden in tmp_file:
                        
                        tamp_data=cv2.imread(tmp_file,0)
                        back=np.array(tamp_data)-np.array(img_data)
                        back[back!=0]=np.array(tamp_data)[back!=0]
                        idx_box = np.where(back!=0)
                        y,h,x,w = np.min(idx_box[0]), np.max(idx_box[0]), np.min(idx_box[1]), np.max(idx_box[1])
                        template = np.array(tamp_data)[y:h,x:w]
                        cv2.imwrite(os.path.join(self.tmpl_dir,'{}_template.png'.format(iden)),template)
            # Saves the template->
                tmplt_file=os.path.join(self.tmpl_dir,'{}_template.png'.format(iden))
                template_arr=cv2.imread(tmplt_file,0)
                w, h = template_arr.shape[:2]
                for id_s in idx:
                    #save img 
                    tmp_file=self.tampered_files[id_s]
                    image_data=cv2.imread(tmp_file)
                    image_data=cv2.resize(image_data,(self.image_dim,self.image_dim))
                    cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),image_data)
                    
                    # create ground truth
                    tamp_data=cv2.imread(tmp_file,0)
                    back=tamp_data-img_data
                    back[back!=0]=255
                    # similiarity data
                    res = cv2.matchTemplate(img_data,template_arr,5)
                    _,_,_,top_left = cv2.minMaxLoc(res)
                    back[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=255
                    back=cv2.resize(back,(self.image_dim,self.image_dim),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),back)
                    self.data_count+=1      
    def prepare(self):
        self.__renameProblematicFile()
        self.__listFiles()
        self.__create_ds()


class MICC_F220(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='MICC_F220'):
        self.data_dir=data_dir

        self.save_dir=os.path.join(save_dir,'DataSets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,iden)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        

        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'targets')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)


        self.tmpl_dir=os.path.join(self.save_dir,'Templates')
        if not os.path.exists(self.tmpl_dir):
            os.mkdir(self.tmpl_dir)
        
        self.image_dim=image_dim
        # Dataset Specific
        self.tamper_iden='tamp'
        self.orig_iden='_scale'
        self.base_tamp_iden='tamp1.jpg'
        self.data_count=0
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def __listFiles(self):
        self.tampered_files=[]
        self.tamper_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            self.tamper_idens.append(base_name)
            self.tampered_files.append(file_name)
            
        self.original_files=[]
        self.orig_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            self.orig_idens.append(base_name)
            self.original_files.append(file_name)
        


    def __create_ds(self):
        for iden in tqdm(self.orig_idens):
            #<-Saves the templates
            if iden in self.tamper_idens:
                base_image=self.original_files[int(self.orig_idens.index(iden))]
                img_data=cv2.imread(base_image,0)
                idx=[i for i, e in enumerate(self.tamper_idens) if e == iden] 
                for id_s in idx:
                    tmp_file=self.tampered_files[id_s]
                    if self.base_tamp_iden in tmp_file:
                        
                        tamp_data=cv2.imread(tmp_file,0)
                        back=np.array(tamp_data)-np.array(img_data)
                        back[back!=0]=np.array(tamp_data)[back!=0]
                        idx_box = np.where(back!=0)
                        y,h,x,w = np.min(idx_box[0]), np.max(idx_box[0]), np.min(idx_box[1]), np.max(idx_box[1])
                        template = np.array(tamp_data)[y:h,x:w]
                        cv2.imwrite(os.path.join(self.tmpl_dir,'{}_template.png'.format(iden)),template)
            # Saves the template->
                tmplt_file=os.path.join(self.tmpl_dir,'{}_template.png'.format(iden))
                template_arr=cv2.imread(tmplt_file,0)
                w, h = template_arr.shape[:2]
                for id_s in idx:
                    #save img 
                    tmp_file=self.tampered_files[id_s]
                    image_data=cv2.imread(tmp_file)
                    image_data=cv2.resize(image_data,(self.image_dim,self.image_dim))
                    cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),image_data)
                    
                    # create ground truth
                    tamp_data=cv2.imread(tmp_file,0)
                    back=tamp_data-img_data
                    back[back!=0]=255
                    # similiarity data
                    res = cv2.matchTemplate(img_data,template_arr,5)
                    _,_,_,top_left = cv2.minMaxLoc(res)
                    back[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=255
                    back=cv2.resize(back,(self.image_dim,self.image_dim),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),back)
                    self.data_count+=1    
    def prepare(self):
        self.__listFiles()
        self.__create_ds()
                    
        
class MICC_F600(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='MICC_F600'):
        self.data_dir=data_dir

        self.save_dir=os.path.join(save_dir,'DataSets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,iden)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'targets')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)

        self.image_dim=image_dim
        self.data_count=0
        self.gt_iden='_gt'
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def prepare(self):
    
        for file_name in tqdm(glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))):
            # Ground Truth Saving
            gt=cv2.imread(file_name,0)
            gt=cv2.resize(gt,(self.image_dim,self.image_dim))
            ## Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gt,(5,5),0)
            _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),gt)
            #image
            base_path,fmt=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.gt_iden)]
            img_path=os.path.join(self.data_dir,'{}{}'.format(base_name,fmt))
            img=cv2.imread(img_path)
            img=cv2.resize(img,(self.image_dim,self.image_dim))
            cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),img)
            self.data_count+=1
                
class CoMoFoD(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='CoMoFoD'):
        self.data_dir=data_dir

        self.save_dir=os.path.join(save_dir,'DataSets')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,iden)
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'targets')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)

        self.image_dim=image_dim
        self.data_count=0
        self.gt_iden='_B'
        self.im_iden='_F'
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def prepare(self):
    
        for file_name in tqdm(glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))):
            # Ground Truth Saving
            gt=cv2.imread(file_name,0)
            gt=cv2.resize(gt,(self.image_dim,self.image_dim))
            ## Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gt,(5,5),0)
            _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),gt)
            #image
            img_path=str(file_name).replace(self.gt_iden,self.im_iden)
            img=cv2.imread(img_path)
            img=cv2.resize(img,(self.image_dim,self.image_dim))
            cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),img)
            self.data_count+=1
            

    
   
