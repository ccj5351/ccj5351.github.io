---
layout: page
title: "Stereo Matching Datasets Info"
comments: false
date: 2018-12-28
---

## Datasets

The popular datasets for stereo matching include:

- synthetic datasets: e.g., Scene Flow dataset;
- Real world datasets: e.g, Middlebury V3, KITTI 2012&2015, ETH3D.

## Statistics Info

- Scene Flow: $\approx$ 39k stereo frames in $960 \times 540$ pixel resolution.
 - Total Image # : 35454;
 - Image # with disparity $\geq 256$ : 6555, i.e., $ 6555 / 35454 = 18.49%$

- KITTI : KITTI 12 and KITTI 15, roughtly with size $1240 \times 388$.
 - Disparity : $d \leq 255$
- ETH3D : See the following table-1. According to its website, no information about the disparity range is provided: the 
corresponding field is set to the **image width**.
- Middlebury V3: see the following table-2.


## Middlebury V3 Info :

- trainingH:

| Images (including 15 images)   | Height | Width | Max Disparity | vmin | vmax |
| :-------  |:-----: |:----: | :-----------: |:----:|:----:|
|Adirondack |  992   | 1436  |    145	       |  16  |  129 |
|ArtL       |  554   | 694   |	  128	       |  32  |  115 |
|Jadeplant  | 994    | 1318  |    320	       |  11  |  298 |
|Motorcycle | 994    | 1482  |	  140        |	15  |  129 |
|MotorcycleE| 994    | 1482  |    140        |	15  |  129 |
|Piano      | 962    | 1414  |    130	       |  18  |  109 |
|PianoL     | 962    | 1414  |    130	       | 18   |  109 |
|Pipes      | 970    | 1470  |	  150	       |13    |  138 |
|Playroom   | 952    | 1398  |	  165	       |15    |	 151 |
|Playtable  | 926    | 1360  |	  145	       |13    |	 136 |
|PlaytableP | 924    | 1362  |	  145	       |13    |	 135 |
|Recycle    | 972    | 1440  |	  130				 |16    |	 114 |
|Shelves    | 994    | 1476  |	  120				 |26    |	 103 |
|Teddy      | 750    | 900   |	  128				 |25    |	 110 |
|Vintage    | 960    | 1444  |	  380				 |17    |	 362 |

## ETH3D Info:

- training data:

| Images (including 27 images) | Height | Width | Max Disparity |
| :----------------- |:-----: |:----: | :-----------: |
| delivery_area_1s   | 435    |   711 |     711       |
| electro_1l         | 	489   |	  927 | 	927   |
| electro_2s         | 	435|	717|	717|
| facade_1s          | 	425|	707|	707|
| forest_1s          | 	441|	715|	715|
| forest_2s          | 	440|	715|	715|
| playground_1l| 	490|	941|	941|
| playground_1s| 	436|	712|	712|
| playground_2l| 	490|	941|	941|
| playground_3l| 	490|	941|	941|
| playground_3s| 	436|	712|	712|
| terrains_1l| 	491|	940|	940|
| terrains_1s| 	438|	713|	713|
| terrains_2l| 	491|	940|	940|
| terrains_2s| 	438|	713|	713|
| playground_2s| 	436|	712|	712|
| delivery_area_3l| 	489|	942|	942|
| delivery_area_1l| 	489|	942|	942|
| delivery_area_3s| 	435|	711|	711|
| terrace_1s| 	434|	713|	713|
| terrace_2s| 	434|	713|	713|
| electro_2l| 	489|	927|	927|
| delivery_area_2l| 	489|	942|	942|
| delivery_area_2s| 	435|	711|	711|
| electro_1s| 	435|	717|	717|
| electro_3s| 	435|	717|	717|
| electro_3l| 	489|	927|	927|
