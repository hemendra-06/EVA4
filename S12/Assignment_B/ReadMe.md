## Assignment B

### JSON File Description(coco format)
The coco format of the json file generated after annotating in using the VGG annotator looks like the one showed below: 

    "images": 
    [{
            "id": 0, #id assigned to each image
            "width": 1280, #width of the original image
            "height": 960, #height of the original image
            "file_name": "dog_image_1.jpg", #original image name
            "license": 1,
            "date_captured": ""
    }],
    "annotation": 
    [{
            "id" : int,
            "image_id": int,
            "category_id": int,
            "segmentation": RLE or [polygon],
            "area": float,
            "bbox": [x,y,width,height],
            "iscrowd": 0 or 1,
        }],
    "categories": 
    [{
            "id": int,
            "name": str,
    }]


![Image description](https://github.com/hemendra-06/EVA4/blob/master/S12/Assignment_B/images/Elbow_graph.PNG)