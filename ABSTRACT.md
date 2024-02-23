The authors collect a **PIC: Person in Context Dataset**, which contains 17, 605 high-resolution images and densely annotated entity segmentation and relations, including 141 object categories, 23 relation categories. The authors introduced a new task named human-centric relation segmentation (HRS). HRS aims to predict the relations between the human and surrounding entities and identify the relation-correlated human parts, which are represented as pixel-level masks.

**Note:** the authors did not furnish the capability to compare a particular object with its surrounding entities in detail. The annotations merely offer broad descriptions of the interactions within the image. Consequently, in our scenario, the dataset is solely applicable for addressing instance segmentation challenges.

## Dataset description

Data collection was done with three steps:

* Data crawling: PIC contains both ***indoor*** and ***outdoor*** images, which are crawled from [Flickr](https://www.flickr.com/) website with copyrights. Query words for retrieving ***indoor*** pictures include cook, party, drink, watch tv, eat, etc., and those for ***outdoor*** ones include play ball, run, ride, outing, picnic, etc. In this way, the collected data enjoy great diversities in terms of scenario, appearance, viewpoint, light condition and occlusion.
* Data filtering: The authors filter out the images with low resolution or without human. Then they calculate the distributions of the relations. 
* Data balancing: The authors recollect the data for relations with lower frequency to balance the data distribution. 

The authors first annotate 141 kinds of things and stuff in the images. The entity categories cover a wide range of indoor and outdoor scenes, including office,
restaurant, seaside, snowfield, etc. For each entity falling into predefined categories, they label it with its class and pixel-level mask segment. The disconnected regions of stuff are viewed as different entities. 

<img src="https://github.com/dataset-ninja/person-in-context/assets/120389559/c2c04555-35fb-46e1-94f6-45f4b0a74509" alt="image" width="800">

<span style="font-size: smaller; font-style: italic;">An example of the original image and entity segmentation.</span>