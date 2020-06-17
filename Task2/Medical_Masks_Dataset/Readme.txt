The dataset comes from Eden Social Welfare Foundation, Taiwan.

The original dataset contained 678 images with their xml annotations, but the images and xml files were having random/very-long names and many xml files didnt have their corresponding images

I have split the dataset randomly into 670 images for training and 8 images for testing, (if you dont need this kind of separation, just combine these images). Also, I have created the bounding box labels in various formats for easy use.  


For Pascal VOC Annotations, use
Images, xml_labels, test_images

For Monk-type (in this type,image id & its coordinates of bounding box + bounding box labels are stored in csv), use
Images, monk_train_labels.csv, test_images

For coco type, use
Images, annotations, test_images

For Yolo type, use
Images, yolo_labels, classes.txt, test_images 

