# OIRDS dataset car detection

### Dataset:

https://sourceforge.net/projects/oirds/?source=typ_redirect

### instructions:

Download dataset and place it into `data/oirds`

Run `sh convert_to_png.sh` to create PNG images

Run `python3 create_dataset_folders.py 64 False` to create dataset folders

Run `python3 finetune.py` to train a neural network on OIRDS to detect car/no_car patches