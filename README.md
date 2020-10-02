# pyCMFD
Copy Move Image Forgery Detection with Segmentation Models 

    Version: 0.0.3  
    Python : 3.6.9
    Author : Md. Nazmuddoha Ansary 
             Shakir Hossain  
             Mohammad Bin Monjil  
             Habibur Rahman
             Shahriar Prince  

![](/info/src_img/python.ico?raw=true )
![](/info/src_img/tensorflow.ico?raw=true)
![](/info/src_img/keras.ico?raw=true)
![](/info/src_img/col.ico?raw=true)

# Version and Requirements  
* Python == 3.6.9
* pip3 install -r requirements.txt

# Data Set

* **Evaluation**: 
  * [Casia](https://drive.google.com/file/d/1KvF7EF-rLD2e5AujOzOifBvo5dFTZjbn/view)
  * [CoMoFoD](https://www.vcl.fer.hr/comofod/)
* **Training**:
  * [MICC](http://lci.micc.unifi.it/labd/2015/01/copy-move-forgery-detection-and-localization/)
  * [COVERAGE](https://onedrive.live.com/?authkey=%21ADJSupKlX%5FIj8Yc&id=4B518F0277851508%21709&cid=4B518F0277851508)
  * [Grip](http://www.grip.unina.it/research/83-image-forensics/90-copy-move-forgery.html)
  * [CVIP](http://www.diid.unipa.it/cvip/?page_id=48)
  * [FFCMF](http://emregurbuz.tc/research/imagedatasets/ffcmf/ffcmf.html)
* **Synthetic data Generation**
  * [COCO](https://cocodataset.org/#download)

## Processing

* **The folder structure after Dividing CVIP and MICC is as follows**

     

      ​       ├── CASIA
      ​       │   ├── Au
      ​       │   ├── GT_Mask
      ​       │   └── Tp
      ​       ├── coco
      ​       │   ├── annotations
      ​       │   └── train2014
      ​       ├── CoMoFoD
      ​       ├── COVERAGE
      ​       │   ├── image
      ​       │   ├── label
      ​       │   └── mask
      ​       ├── CVIP0
      ​       ├── CVIP12
      ​       │   ├── im1
      ​       │   ├── im10
      ​       │   ├── im11
      ​       │   ├── im12
      ​       │   ├── im13
      ​       │   ├── im14
      ​       │   ├── im15
      ​       │   ├── im16
      ​       │   ├── im17
      ​       │   ├── im18
      ​       │   ├── im19
      ​       │   ├── im2
      ​       │   ├── im20
      ​       │   ├── im3
      ​       │   ├── im4
      ​       │   ├── im5
      ​       │   ├── im6
      ​       │   ├── im7
      ​       │   ├── im8
      ​       │   └── im9
      ​       ├── F2000
      ​       ├── F220
      ​       ├── F600
      ​       ├── FFCMF
      ​       └── GRIP

* **dataset.py** holds all the code for processing the datasets

* **data.ipynb** creates the Train and Eval Dataset (in TFRECORD and Image format locally)

**ENVIRONMENT**

        OS          : Ubuntu 18.04.3 LTS (64-bit) Bionic Beaver     
        Memory      : 7.7 GiB  
        Processor   : Intel® Core™ i5-8250U CPU @ 1.60GHz × 8    
        Graphics    : Intel® UHD Graphics 620 (Kabylake GT2)  
        Gnome       : 3.28.2 