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
import shutil
###########################################################################################################################################################

def create_dir(base,path):
    new_dir=os.path.join(base,path)
    if not os.path.exists(new_dir):
        os.mkdir(new_dir)
    return new_dir

def create_struct(save_dir):
    dataset_dir=create_dir(save_dir,'DataSet')
    train_dir=create_dir(dataset_dir,'Train')
    eval_dir=create_dir(dataset_dir,'Eval')
    
    train_img_dir,train_msk_dir=create_img_msk_dir(train_dir)
    eval_img_dir,eval_msk_dir=create_img_msk_dir(eval_dir)
    
    
    dir_dict={'train_dir':train_dir,'train_img_dir':train_img_dir,'train_msk_dir':train_msk_dir,
              'eval_dir':eval_dir,'eval_img_dir':eval_img_dir,'eval_msk_dir':eval_msk_dir}
    return dir_dict
    

def create_img_msk_dir(mode_dir):
    img_dir=create_dir(mode_dir,'images')
    msk_dir=create_dir(mode_dir,'masks')
    return img_dir,msk_dir


###########################################################################################################################################################

class F2000(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='F2000'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)
        self.tmpl_dir=create_dir(os.getcwd(),'Templates')
        self.image_dim=image_dim
        # Dataset Specific
        self.tamper_iden='tamp'
        self.orig_iden='_scale'
        self.base_tamp_iden='tamp1.jpg'
        self.prb_idens=['P1000231','DSCN47']
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
        self.tamper_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            if base_name not in self.prb_idens:
                self.tamper_idens.append(base_name)
                
            
        self.orig_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            if base_name not in self.prb_idens:
                self.orig_idens.append(base_name)
        
        self.all_data_idens=[]
        for iden in self.orig_idens:
            if iden in self.tamper_idens:
                self.all_data_idens.append(iden)

        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
        
    def __mode_ds(self,idens,img_dir,gt_dir,mode):
        
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:    
            for iden in tqdm(idens):
                base_image_path=os.path.join(self.data_dir,'{}{}.jpg'.format(iden,self.orig_iden))
                img_data=cv2.imread(base_image_path,0)
                tamp_file_path=os.path.join(self.data_dir,'{}{}'.format(iden,self.base_tamp_iden))        
                tamp_data=cv2.imread(tamp_file_path,0)
                back=np.array(tamp_data)-np.array(img_data)
                back[back!=0]=np.array(tamp_data)[back!=0]
                idx_box = np.where(back!=0)
                y,h,x,w = np.min(idx_box[0]), np.max(idx_box[0]), np.min(idx_box[1]), np.max(idx_box[1])
                template = np.array(tamp_data)[y:h,x:w]
                cv2.imwrite(os.path.join(self.tmpl_dir,'{}_template.png'.format(iden)),template)
                tmplt_file=os.path.join(self.tmpl_dir,'{}_template.png'.format(iden))
                template_arr=cv2.imread(tmplt_file,0)
                h, w = template_arr.shape[:2]
                # Saves the template->
                for tmp_file in glob(os.path.join(self.data_dir,'{}{}*.jpg'.format(iden,self.tamper_iden))):
                    image_data=cv2.imread(tmp_file)
                    image_data=cv2.resize(image_data,(self.image_dim,self.image_dim))
                    
                    
                    cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),image_data)
                    # create ground truth
                    tamp_data=cv2.imread(tmp_file,0)
                    back=tamp_data-img_data
                    back[back!=0]=255
                    # similiarity data
                    res = cv2.matchTemplate(img_data,template_arr,5)
                    _,_,_,top_left = cv2.minMaxLoc(res)
                    back[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=255
                    back=cv2.resize(back,(self.image_dim,self.image_dim),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),back)
                    data_count+=1      



    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
        
        
    def prepare(self):
        self.__renameProblematicFile()
        self.__listFiles()
        self.__create_ds()
        shutil.rmtree(self.tmpl_dir)

###########################################################################################################################################################



class F220(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='F220'):
        self.data_dir=data_dir

        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)
        self.tmpl_dir=create_dir(os.getcwd(),'Templates')
        self.image_dim=image_dim
        # Dataset Specific
        self.tamper_iden='tamp'
        self.orig_iden='_scale'
        self.base_tamp_iden='tamp1.jpg'
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def __listFiles(self):
        self.tamper_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.tamper_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.tamper_iden)]
            self.tamper_idens.append(base_name)
                
            
        self.orig_idens=[]

        for file_name in glob(os.path.join(self.data_dir,'*{}*.jpg'.format(self.orig_iden))):
            base_path,_=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=base_name[:base_name.find(self.orig_iden)]
            self.orig_idens.append(base_name)
        
        self.all_data_idens=[]
        for iden in self.orig_idens:
            if iden in self.tamper_idens:
                self.all_data_idens.append(iden)

         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
        
        


    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:    
            for iden in tqdm(idens):
                base_image_path=os.path.join(self.data_dir,'{}{}.jpg'.format(iden,self.orig_iden))
                img_data=cv2.imread(base_image_path,0)
                tamp_file_path=os.path.join(self.data_dir,'{}{}'.format(iden,self.base_tamp_iden))        
                tamp_data=cv2.imread(tamp_file_path,0)
                back=np.array(tamp_data)-np.array(img_data)
                back[back!=0]=np.array(tamp_data)[back!=0]
                idx_box = np.where(back!=0)
                y,h,x,w = np.min(idx_box[0]), np.max(idx_box[0]), np.min(idx_box[1]), np.max(idx_box[1])
                template = np.array(tamp_data)[y:h,x:w]
                cv2.imwrite(os.path.join(self.tmpl_dir,'{}_template.png'.format(iden)),template)
                tmplt_file=os.path.join(self.tmpl_dir,'{}_template.png'.format(iden))
                template_arr=cv2.imread(tmplt_file,0)
                h, w = template_arr.shape[:2]
                # Saves the template->
                for tmp_file in glob(os.path.join(self.data_dir,'{}{}*.jpg'.format(iden,self.tamper_iden))):
                    image_data=cv2.imread(tmp_file)
                    image_data=cv2.resize(image_data,(self.image_dim,self.image_dim))
                    cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),image_data)
                    # create ground truth
                    tamp_data=cv2.imread(tmp_file,0)
                    back=tamp_data-img_data
                    back[back!=0]=255
                    # similiarity data
                    res = cv2.matchTemplate(img_data,template_arr,5)
                    _,_,_,top_left = cv2.minMaxLoc(res)
                    back[top_left[1]:top_left[1]+h,top_left[0]:top_left[0]+w]=255
                    back=cv2.resize(back,(self.image_dim,self.image_dim),fx=0, fy=0, interpolation = cv2.INTER_NEAREST)
                    cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),back)
                    data_count+=1      

    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
         
        

    def prepare(self):
        self.__listFiles()
        self.__create_ds()
        shutil.rmtree(self.tmpl_dir)

###########################################################################################################################################################
    
class F600(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='F600'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)
        
        self.image_dim=image_dim
        self.gt_iden='_gt'
        print(colored('Initializing:{}'.format(iden),'green'))
        
    def __listFiles(self):
        self.all_data_idens= [file_name for file_name in glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))]
         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
        
    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for file_name in tqdm(idens):
                # Ground Truth Saving
                gt=cv2.imread(file_name,0)
                gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                ## Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(gt,(5,5),0)
                _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                #image
                base_path,fmt=os.path.splitext(file_name)
                base_name=os.path.basename(base_path)
                base_name=base_name[:base_name.find(self.gt_iden)]
                img_path=os.path.join(self.data_dir,'{}{}'.format(base_name,fmt))
                img=cv2.imread(img_path)
                img=cv2.resize(img,(self.image_dim,self.image_dim))
                 
                cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                data_count+=1    
            


    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
         
        


    def prepare(self):
        self.__listFiles()
        self.__create_ds()
        


###########################################################################################################################################################
          

class GRIP(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='GRIP'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)

        self.image_dim=image_dim
        self.gt_iden='_gt'
        self.im_iden='_copy'
        print(colored('Initializing:{}'.format(iden),'green'))
    
    def __listFiles(self):
        self.all_data_idens= [file_name for file_name in glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))]
         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens

    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for file_name in tqdm(idens):
                # Ground Truth Saving
                gt=cv2.imread(file_name,0)
                gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                ## Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(gt,(5,5),0)
                _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                #image
                base_path,fmt=os.path.splitext(file_name)
                base_name=os.path.basename(base_path)
                base_name=base_name[:base_name.find(self.gt_iden)]
                img_path=str(file_name).replace(self.gt_iden,self.im_iden)
                img=cv2.imread(img_path)
                img=cv2.resize(img,(self.image_dim,self.image_dim))
                 
                cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                data_count+=1    
            


    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
         


    def prepare(self):
        self.__listFiles()
        self.__create_ds()    

###########################################################################################################################################################

class CVIP0(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='CVIP0'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)

        self.image_dim=image_dim
        self.im_iden=''
        self.gt_iden='_mask'
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def __listFiles(self):
        self.all_data_idens= [file_name for file_name in glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))]
         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
    
    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for file_name in tqdm(idens):
                # Ground Truth Saving
                gt=cv2.imread(file_name,0)
                gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                ## Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(gt,(5,5),0)
                _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                #image
                base_path,fmt=os.path.splitext(file_name)
                base_name=os.path.basename(base_path)
                base_name=base_name[:base_name.find(self.gt_iden)]
                img_path=str(file_name).replace(self.gt_iden,self.im_iden)
                img=cv2.imread(img_path)
                img=cv2.resize(img,(self.image_dim,self.image_dim))
                 
                cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                data_count+=1    
            


    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
         
        


    def prepare(self):
        self.__listFiles()
        self.__create_ds()

###########################################################################################################################################################

 
class FFCMF(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='FFCMF'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)

        self.image_dim=image_dim
        self.im_iden='_N'
        self.gt_iden='_M'
        print(colored('Initializing:{}'.format(iden),'green'))
        

    def __listFiles(self):
        self.all_data_idens= [file_name for file_name in glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))]
         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
        
    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for file_name in tqdm(idens):
                # Ground Truth Saving
                gt=cv2.imread(file_name,0)
                gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                ## Otsu's thresholding after Gaussian filtering
                blur = cv2.GaussianBlur(gt,(5,5),0)
                _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                #image
                base_path,fmt=os.path.splitext(file_name)
                base_name=os.path.basename(base_path)
                base_name=base_name[:base_name.find(self.gt_iden)]
                img_path=str(file_name).replace(self.gt_iden,self.im_iden)
                img=cv2.imread(img_path)
                img=cv2.resize(img,(self.image_dim,self.image_dim))
                 
                cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                data_count+=1    
            


    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
         


    def prepare(self):
        self.__listFiles()
        self.__create_ds()

###########################################################################################################################################################
     
class CVIP12(object):
    def __init__(self,data_dir,save_dir,image_dim=256,iden='CVIP12'):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)

        self.image_dim=image_dim
        self.gt_iden='_mask'
        print(colored('Initializing:{}'.format(iden),'green'))
        
    def __listFiles(self):
        self.all_data_idens= [folder_name for folder_name in os.listdir(self.data_dir)]
         
        random.shuffle(self.all_data_idens)
        self.train_idens=self.all_data_idens
        
    def __mode_ds(self,idens,img_dir,gt_dir,mode):
          
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for iden in tqdm(idens):
                for file_name in glob(os.path.join(self.data_dir,iden,'out_*{}.*'.format(self.gt_iden))):
                    # Ground Truth Saving
                    gt=cv2.imread(file_name,0)
                    gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                    ## Otsu's thresholding after Gaussian filtering
                    blur = cv2.GaussianBlur(gt,(5,5),0)
                    _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                    cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                    #image
                    base_path,fmt=os.path.splitext(file_name)
                    base_name=os.path.basename(base_path)
                    base_name=base_name[:base_name.find(self.gt_iden)]
                    img_path=os.path.join(self.data_dir,os.path.basename(os.path.dirname(base_path)),'{}{}'.format(base_name,fmt))
                    img=cv2.imread(img_path)
                    img=cv2.resize(img,(self.image_dim,self.image_dim))
                     
                    cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                    data_count+=1
                

    def __create_ds(self):
        self.__mode_ds(self.train_idens,self.dir_dict['train_img_dir'],self.dir_dict['train_msk_dir'],'train')
          
        


    def prepare(self):
        self.__listFiles()
        self.__create_ds()


###########################################################################################################################################################
class COVERAGE(object):
    def __init__(self,data_dir,save_dir,image_dim=256):
        self.data_dir=data_dir
        self.save_dir=os.path.join(save_dir,'DataSet')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,'Train')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'masks')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)

        self.image_dim=image_dim

        existing=[_path for _path in glob(os.path.join(self.img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        
        self.data_count=data_count        

        self.im_iden='t'
        


    def prepare(self):
        idens=[img_path for img_path in glob(os.path.join(self.data_dir,'image','*{}.*'.format(self.im_iden)))]
        for file_name in tqdm(idens):
            #image
            img=cv2.imread(file_name)
            img=cv2.resize(img,(self.image_dim,self.image_dim))
             
            cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),img)
            # Ground Truth Saving
            base_path,fmt=os.path.splitext(file_name)
            base_name=os.path.basename(base_path)
            base_name=str(base_name).replace(self.im_iden,'')
            cpy=cv2.imread(os.path.join(self.data_dir,'mask','{}copy.tif'.format(base_name)),0)
            frg=cv2.imread(os.path.join(self.data_dir,'mask','{}forged.tif'.format(base_name)),0)
            gt=cpy+frg
            gt=cv2.resize(gt,(self.image_dim,self.image_dim))
            ## Otsu's thresholding after Gaussian filtering
            blur = cv2.GaussianBlur(gt,(5,5),0)
            _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),gt)
            self.data_count+=1
###########################################################################################################################################################

class CoMoFoD(object):
    def __init__(self,data_dir,save_dir,image_dim=256):
        self.data_dir=data_dir
        self.dir_dict=create_struct(save_dir)

        self.image_dim=image_dim
        self.gt_iden='_B'
       
        
    def __listFiles(self):
        self.all_data_idens= [os.path.basename(file_name) for file_name in glob(os.path.join(self.data_dir,'*{}.*'.format(self.gt_iden)))]
        self.all_data_idens= [str(filename).replace('_B.png','') for filename in self.all_data_idens]
         
        random.shuffle(self.all_data_idens)
        self.eval_idens=self.all_data_idens

    def __mode_ds(self,idens,img_dir,gt_dir,mode):  
        existing=[_path for _path in glob(os.path.join(img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        if idens is not None:
            for iden in tqdm(idens):
                for filename in glob(os.path.join(self.data_dir,'{}_*.*'.format(iden))):
                    if '_F' in filename:
                        # Ground Truth Saving
                        gt_file=os.path.join(self.data_dir,'{}{}.png'.format(iden,self.gt_iden))
                        gt=cv2.imread(gt_file,0)
                        gt=cv2.resize(gt,(self.image_dim,self.image_dim))
                        ## Otsu's thresholding after Gaussian filtering
                        blur = cv2.GaussianBlur(gt,(5,5),0)
                        _,gt = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        cv2.imwrite(os.path.join(gt_dir,'{}.png'.format(data_count)),gt)
                        #image
                        img=cv2.imread(filename)
                        img=cv2.resize(img,(self.image_dim,self.image_dim))
                        cv2.imwrite(os.path.join(img_dir,'{}.png'.format(data_count)),img)
                        data_count+=1    

    def __create_ds(self):
        self.__mode_ds(self.eval_idens,self.dir_dict['eval_img_dir'],self.dir_dict['eval_msk_dir'],'eval')
          
        


    def prepare(self):
        self.__listFiles()
        self.__create_ds()          
        
###########################################################################################################################################################
class CASIA(object):
    def __init__(self,data_dir,save_dir,image_dim=256):
        self.data_dir=data_dir
        self.save_dir=os.path.join(save_dir,'DataSet')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)
        
        self.save_dir=os.path.join(self.save_dir,'Eval')
        if not os.path.exists(self.save_dir):
            os.mkdir(self.save_dir)

        self.img_dir=os.path.join(self.save_dir,'images')
        if not os.path.exists(self.img_dir):
            os.mkdir(self.img_dir)

        self.gt_dir=os.path.join(self.save_dir,'masks')
        if not os.path.exists(self.gt_dir):
            os.mkdir(self.gt_dir)

        self.image_dim=image_dim
        
        existing=[_path for _path in glob(os.path.join(self.img_dir,"*.*"))]
        if existing is None:
            data_count=0
        else:
            data_count=len(existing)
        
        self.data_count=data_count         

        


    def prepare(self):
        idens=[img_path for img_path in glob(os.path.join(self.data_dir,'GT_Mask','*.*'))]
        for file_name in tqdm(idens):
            img_iden=str(file_name).replace('GT_Mask','Tp')
            img_iden=img_iden.replace('.png','')
            #image
            for img_path in glob(os.path.join(self.data_dir,'Tp','{}.*'.format(img_iden))):
                img=cv2.imread(img_path)
                img=cv2.resize(img,(self.image_dim,self.image_dim))
                cv2.imwrite(os.path.join(self.img_dir,'{}.png'.format(self.data_count)),img)
                break
            # Ground Truth Saving
            gt_3=img=cv2.imread(file_name)
            gt_3=cv2.cvtColor(gt_3,cv2.COLOR_BGR2RGB)
            gt_b=gt_3[:,:,2]
            gt=np.ones((gt_b.shape[0],gt_b.shape[1]))*255
            gt[gt_b==255]=0
            gt=cv2.resize(gt,(self.image_dim,self.image_dim),fx=0,fy=0,interpolation = cv2.INTER_NEAREST)
            cv2.imwrite(os.path.join(self.gt_dir,'{}.png'.format(self.data_count)),gt)
            self.data_count+=1
###########################################################################################################################################################

