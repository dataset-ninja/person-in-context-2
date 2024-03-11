**PIC: Person in Context Dataset v2.0** is a dataset for instance segmentation, semantic segmentation, and object detection tasks. It is applicable or relevant across various domains. 

The dataset consists of 17605 images with 191150 labeled objects belonging to 144 different classes including *human*, *table*, *bag*, and other: *hat*, *ground*, *chair*, *door*, *painting/poster*, *sofa*, *building*, *shelf*, *window*, *grass*, *cabinet*, *vegetation*, *floor*, *guardrail*, *ball*, *book*, *curtain*, *cup*, *phone*, *bottle*, *toy*, *tree*, *plant*, *stick*, *instrument*, and 116 more.

Images in the Person in Context dataset have pixel-level instance segmentation annotations. Due to the nature of the instance segmentation task, it can be automatically transformed into a semantic segmentation (only one mask for every class) or object detection (bounding boxes for every object) tasks. There are 2975 (17% of the total) unlabeled images (i.e. without annotations). There are 3 splits in the dataset: *train* (12653 images), *test* (2975 images), and *val* (1977 images). Alternatively, the dataset could be split into 2 location: ***indoor*** (7458 images) and ***outdoor*** (6539 images). The dataset was released in 2021 by the <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Beihang University, China</span>, <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Academy of Science, China</span>, and <span style="font-weight: 600; color: grey; border-bottom: 1px dashed #d3d3d3;">Sea AI Lab, China</span>.

Here is a visualized example for randomly selected sample classes:

[Dataset classes](https://github.com/dataset-ninja/person-in-context-2/raw/main/visualizations/classes_preview.webm)
