# Sign2Sign Transformer

## Install
Install required packages using the requirements.txt file.

`pip install -r requirements (1).txt`

## Usage

To run, start __main__.py with arguments "train" and ".\Configs\Base.yaml":

`python __main__.py train ./Configs/Base.yaml` 

You need ot modify the yaml file.

(1) You can use different versions of PT dataset by modifying the data path (train, dev, test and src_vocab path). (located under data folder)


### Dump the skeletons

First call the below function in `training.py` main function by modifying lines 687 and beyond.

`test(cfg_file=PATH_OF_MODEL_CONFIG_FILE, ckpt=get_latest_checkpoint( ATH_OF_TRAINED_MODEL,post_fix="_best"))


Then `python training.py` 

### Convert Data to SLT format.

`python write_file_slt.py --cur_path Data/MODEL_SKELETON_NAME`

### Model checkpoints 
You can find the model checkpoints in [this link.](https://drive.google.com/drive/folders/10fsw-xSt2Rmupx31FYhxPcfSfiFL5Djf?usp=sharing)

# SLT

Make sure you have the corresponding data.

`cd slt`

`pip install -r requirements.txt`

## Training
`python -m signjoey test configs/YOUR_MODIFIED_YAML`
you only need to modify the data:data_path in the YAML file.

## Test
`python -m signjoey test configs/YOUR_MODIFIED_YAML --ckpt your.ckpt --output_path output/MODEL_NAME`


