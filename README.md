# Data-Mining

<p align="center">
    <img src="./imgs/fusionride - logo.jpeg" width="600px">
</p>


## The repository for the tag discovery in the autonomous vehicle (AV) data using CLIP model, a pre-trained foundation model. 

This is part of Fusionride AI toolkit. In order to start using this repository, few steps are needed to be taken:


### 1. Installing the required packages

The repository contains ```requirements.txt``` file with all the necessary packages. These can be installed with the following:

```bash
pip install -r requirements.txt 

```

These packages can also be installed manually one by one if there are any problems.


### 2. Configuration

The project relies on parquet files for a faster loading. Therefore, it is necessary to configure paths of data that we want to use and store them in the ```./config/config_files``` directory.
To do that, run: 

```bash

./config.sh

```
The script will ask for the number of folders where you store data. The structure of folders used for that should look as follows:

```
├── data
│   └── img001.jpg
    └── img002.jpg
    └── img003.jpg
    └── ...

```

Hence, please adjust your folders accordingly before configuration. Next, the script will ask for a full path to each directory with data in one-by-one fashion. What's important, these directories have to have different names otherwise the script will overwrite the configuration files. Additionally, if you move your data directories, a reconfiguration will be necessary. However, this can be easily done with the aforementioned shell file and overwriting the current parquet file that needs to be configured.


### 3. Running the main code

Finally, to run the main script, use:

```bash

./run.sh

```

Firstly, the script will create three requests to be defined by a user:
    1. The caption which will be matched to configured images.
    2. The top-k images with a highest score to be visualised.
    3. The number of images to be used for the whole run.

After defining these, the model will process for a while. Make sure a CUDA GPU support is available for a computation acceleration. The program provides an information whether it runs on CPU or CUDA GPU. After the execution is finished, the visualisations are saved into the ```./output visualisations``` folder. 


