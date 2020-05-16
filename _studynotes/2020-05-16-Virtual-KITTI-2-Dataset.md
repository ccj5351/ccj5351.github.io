---
layout: page
title: "Virtual KITTI 2 Dataset"
date : 2020-05-16
---

> see: https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/

[Virtual KITTI 2](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/) is a more photo-realistic and better-featured version of the original virtual KITTI dataset. It exploits recent improvements of the Unity game engine and provides new data such as stereo images or scene flow.

## 1. Dataset Statistics

The Virtual KITTI 2 dataset has $21,260$ stereo pairs in total, including (where each scene represents a location):
 - Scene01: $4,470$, crowded urban area; 
 - Scene02: $2,330$, road in urban area then busy intersection;
 - Scene06: $2,700$, stationary camera at a busy intersection;
 - Scene18: $3,390$, long road in the forest with challenging imaging conditions and shadows;
 - Scene20: $8,370$, highway driving scene.

## 2. Format Description

> SceneX/Y/frames/rgb/Camera_Z/rgb_%05d.jpg
> SceneX/Y/frames/depth/Camera_Z/depth_%05d.png
> SceneX/Y/frames/classsegmentation/Camera_Z/classgt_%05d.png
> SceneX/Y/frames/instancesegmentation/Camera_Z/instancegt_%05d.png
> SceneX/Y/frames/backwardFlow/Camera_Z/backwardFlow_%05d.png
> SceneX/Y/frames/backwardSceneFlow/Camera_Z/backwardSceneFlow_%05d.png
> SceneX/Y/frames/forwardFlow/Camera_Z/flow_%05d.png
> SceneX/Y/frames/forwardSceneFlow/Camera_Z/sceneFlow_%05d.png
> SceneX/Y/colors.txt
> SceneX/Y/extrinsic.txt
> SceneX/Y/intrinsic.txt
> SceneX/Y/info.txt
> SceneX/Y/bbox.txt
> SceneX/Y/pose.txt

> where <strong> X ∈ {01, 02, 06, 18, 20}</strong> and represent one of 5 different locations.
> <strong> Y ∈ {15-deg-left, 15-deg-right, 30-deg-left, 30-deg-right, clone, fog, morning, overcast, rain, sunset}</strong> and represent the different variations.
> <strong> Z ∈ [0, 1]</strong> and represent the left (same as in virtual kitti) or right camera (offset by <strong>0.532725m</strong> to the right). 
> Note that our indexes always start from 0.

## 3. Depth $z$
All depth images are encoded as Grayscale 16bit PNG files.We use a fixed far plane of $655.35$ meters (pixels farther away are clipped; however, it is not relevant for this dataset). 
This allows us to truncate and normalize the $z$ values to the $[0, 2^{16} – 1]$ integer range such that a pixel intensity of $1$ in our single channel PNG16 depth images corresponds 
to a distance of $1$ cm to the camera plane.The depth map in `centimeters` can be directly loaded in Python with numpy and OpenCV via the one-liner (assuming “import cv2”):

```python
depth = cv2.imread(depth_png_filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

```

## 4. Disparity $d$

Based on $d = fB/z$, we can get the disparity $d$. BY checking the `intrinsic.txt` files, I find $f = 725.0087$, $B = 53.2725$ (cm), and the depth png files contain the $z$ values in centimeters. 
So to get the disparity maps, I just run $d=fB/z$ ( since I find no zero-value of depth $z$).

For example, we show `Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001.jpg` and `Virtual-KITTI-V2/vkitti_2.0.3_rgb/Scene01/15-deg-left/frames/rgb/Camera_1/rgb_00001.jpg`.

- stereo pair (move mouse on it to see the right view):  
<a href="#" id="Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001"> <img title="Scene01/15-deg-left/frames/rgb/Camera_0/rgb_00001" src="/files/stereo-datasets/virtual-kitti2/left_rgb_00001.jpg" onmouseover="this.src='/files/stereo-datasets/virtual-kitti2/right_rgb_00001.jpg'" onmouseout="this.src='/files/stereo-datasets/virtual-kitti2/left_rgb_00001.jpg'"/></a>  
- disparity:  
<img src= "/files/stereo-datasets/virtual-kitti2/disp_00001.png" alt = "semantic segmentation" class="center"/>  

- class segmentation:  
<img src= "/files/stereo-datasets/virtual-kitti2/classgt_00001.png" alt = "semantic segmentation" class="center"/>  

## 5. Class Segmentation

I find the `colors.txt` file has 15 colors or classes in total.

```python
from collections import namedtuple
# a label and all meta information
Label = namedtuple( 'Label' , [

    'name'        , # The identifier of this label, e.g. 'car', 'person', ... .
                    # We use them to uniquely name a class

    'id'          , # An integer ID that is associated with this label.
                    # The IDs are used to represent the label in ground truth images
                    # An ID of -1 means that this label does not have an ID and thus
                    # is ignored when creating ground truth images (e.g. license plate).
                    # Do not modify these IDs, since exactly these IDs are expected by the
                    # evaluation server.

    'color'       , # The color of this label
    ] )

virtual_kitti_2_labels = [
    #       name                     id        color
    Label(  'undefined'            ,  0 ,      (  0,  0,  0)   ),
    Label(  'terrain'              ,  1 ,      (210,  0,  200) ),
    Label(  'sky'                  ,  2 ,      ( 90, 200, 255) ),
    Label(  'tree'                 ,  3 ,      (  0, 199, 0)   ),
    Label(  'vegetation'           ,  4 ,      ( 90, 240, 0)   ),
    Label(  'building'             ,  5 ,      (140, 140, 140) ),
    Label(  'road'                 ,  6 ,      (100, 60,  100) ),
    Label(  'guard rail'           ,  7 ,      (250, 100, 255) ),
    Label(  'traffic sign'         ,  8 ,      (255, 255, 0)   ),
    Label(  'traffic light'        ,  9 ,      (200, 200, 0)   ),
    Label(  'pole'                 , 10 ,      (255, 130, 0)   ),
    Label(  'misc'                 , 11 ,      (80,  80, 80)   ),
    Label(  'truck'                , 12 ,      (160, 60, 60)   ),
    Label(  'car'                  , 13 ,      (255, 127, 80)  ),
    Label(  'van'                  , 14 ,      (0, 139, 139)   ),
]
```

## 6. Train-Test Split for Stereo Matching Experiments

Maybe there are three plans for train/test split: 
-  1) mix up all of them, then randomly select $3,000$ pairs (i.e., $14\%$) as test set,
-  2) randomly choose $14\%$ from each scene category for testing set, or 
-  3) keep one scene, like <strong>Scene06</strong> as test set.

Which one do you prefer? I find no specification of the train/test split on the official website. 

My solution for stereo matching experiments:

- 1) Hold <strong>Scene06</strong> back, which will not be used in our experiments.
- 2) randomly choose $14\%$ from each scene category among the remaining ones for validation (or test) set.

resulting in a training set of $15,961$ images (see the file list [virtual_kitti2_wo_scene06_train.list]({% link files/stereo-datasets/virtual-kitti2/virtual_kitti2_wo_scene06_train.list %})); and a testing set of $2599$ 
images (see the file list [virtual_kitti2_wo_scene06_test.list]({% link files/stereo-datasets/virtual-kitti2/virtual_kitti2_wo_scene06_test.list %})).