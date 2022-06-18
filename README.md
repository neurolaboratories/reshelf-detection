# CPGDet-129
We're excited to open source one of Neurolabs’ synthetic datasets, **CPGDet-129** that is used in our retail-focused flagship product, ReShelf. Neurolabs is leading the way for synthetic computer vision applied across the value chain of consumer product goods (CPG). **CPGDet-129** been used to train a synthetic computer vision model for the task of object localisation and is the first public synthetic dataset designed specifically for the task of object detection. 

Our [medium post](http://google.com) describes in detail this dataset and some of our learnings. 


## Filtering Annotations 
In order to obtain the best results from this dataset, a post processing script for filtering out annotations is provided. 
This is needed in order to obtain consistent annotations with how a human would label a dataset. 

Filtering can be done based on a combination of the following:

- visible percentage of the whole product, i.e. a bottle might be just 50% visible because it's occluded by a bottle in front of it
- statistics of the bbox distribution for a class: removing all bboxes that are X*std away from the mean
- absolute value of the bbox area, eg remove all bboxes smaller than 145 pixels

Example of how to call it:

`python postprocess.py filter --visible-percentage 0.6 --std-threshold 1.5 --area-threshold 150`

will remove all annotations which have the visibility < 60%, that have the area < mean(class) - 1.5* std(class) and have the absolute area value < 150 pixels

More info 
`python postprocess.py filter -h` 
