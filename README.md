# GAMA


This is the code for the paper "Think Before You Act: A Two-Stage Framework for Mitigating Gender Bias Towards Vision-Language Tasks"

## Environment


```shell
conda create -n gama python=3.8
pip install -r requirements.txt
```

## Step 1: Download Datasets

Please first download the datasets.

+ Localized Narratives for Open Images: https://google.github.io/localized-narratives/
+ MSCOCO: https://cocodataset.org/
+ Flickr30k: https://bryanplummer.com/Flickr30kEntities/
+ VisoGender: https://github.com/oxai/visogender
+ VL-Bias: https://github.com/VL-Bias/VL-Bias



## Step 2: Extract Image Feature

```shell
python extract_features.py --data_root path-to-images --output_dir path-to-output --dataset dataset-name
```

## Step 3: Input Data Format

Please format the datasets and put the data into "./dataset/dataset_name"

### Narrative Generation
```
{
    "id": "03f8fb9315c004e5",  // image_id
    "image_path": "03f8fb9315c004e5.jpg",  // image name
    "1": {
        "context": "",
        "question": "What can you see in this image ? Please describe it in detail . ",  // question prompt
        "answer": "In this picture I can observe a man and woman . Woman is holding a paper in her hand . In the background I can observe a building and plants .",  // ground truth narrative
        "answer_2": "In this picture I can observe a [GENDER] and [GENDER] . [GENDER] is holding a paper in [GENDER] hand . In the background I can observe a building and plants ."  // narrative with masked gender words
    }
}
```

### Image Captioning

```
{
        "id": "57870",  // image_id
        "image_path": "COCO_train2014_000000057870.jpg",  // image name
        "1": {  // data for stage one 
            "context": "",
            "question": "What can you see in this image ? Please describe in detail .",
            "answer": ""
        },
        "2": {
            "context": "",
            "question": "Generate a short caption of the image .",  // question prompt
            "answer": "A restaurant has modern wooden tables and chairs."  // ground_truth answer
        }
    }
```


### Image Search

The test data for VisoGender and VL-Bias should also be converted to this format.

```
{
    "id": "57870",
    "image_path": "COCO_train2014_000000057870.jpg",
    "1": {
        "context": "",
        "question": "What can you see in this image ? Please describe in detail .",
        "answer": ""
    },
    "2": {
        "context": "",
        "question": "Do the image and the query match ?",
        "caption": "A restaurant has modern wooden tables and chairs.",  // search query
        "answer": "yes"   // We only need matched pairs in train.json
    }
}
```



## Step 3: Model Training & Evaluation

Config files with hyper-parameters are available in json files under the "./config"


### Narrative Generation Model

First, you need to train the model for narrative generation and predict narratives for task-specific dataset.

```shell
# train
python main.py --config ./config/train/open_images.json --do_train

# predict on MSCOCO
python main.py --config ./config/pred/mscoco-narrative_generation.json --do_test
```

Please move the prediction file to "./dataset/localized_narratives_pred/" and rename it as the dataset name, like "mscoco.json".

### Answer Inference Model

Then you can train task-specific answer inference models. 

Here is an example for image captioning on MSCOCO.

```shell
# MSCOCO caption

# train
python main.py --config ./config/train/mscoco-caption.json --do_train

# predict on MSCOCO
python main.py --config ./config/pred/mscoco-caption.json --do_test
```
