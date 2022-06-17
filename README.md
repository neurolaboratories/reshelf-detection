# Reshelf Detection

## Filtering Annotations 
Running 
`python src/main_postprocess.py filter --visible-percentage 0.6 --std-threshold 1.5 --area-threshold 150`

will remove all annotations which have the visibility < 60%, that have the area < mean(class) - 1.5* std(class) and have the absolute area value < 150 pixels
