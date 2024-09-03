# Real-Time Instrumental Playing Techniques Recognition

![Status: Not Ready](https://img.shields.io/badge/status-not%20ready-red)

This project focuses on real-time recognition of instrumental playing techniques using advanced machine learning models. It enables the automatic identification of various playing techniques in real time from an audio stream input. This repository includes tools for preparing datasets, training models, evaluating their performance, and real-time inference.

**Lead Developer:** [Nicolas Brochec](https://orcid.org/0009-0000-5922-2079), Tokyo University of the Arts, ERC Reach.

**Contributor:** [Marco Fiorini](https://orcid.org/0009-0004-0523-2481), IRCAM-STMS, CNRS, Sorbonne Université, ERC Reach.

## Installation
Clone this repository and navigate to the folder.
```
git clone https://github.com/nbrochec/realtimeIPTrecognition/
cd realtimeIPTrecognition
```
Create a conda environment with Python 3.11.7
```
conda create --name IPT python=3.11.7
conda activate IPT
```
Install dependencies.
```
pip install -r requirements.txt
```

## Folder structure

```
└── 📁data
    └── 📁dataset
    └── 📁raw_data
        └── 📁test
        └── 📁train
└── 📁externals
    └── 📁pytorch_balanced_sampler
        └── __init__.py
        └── sampler.py
        └── utils.py
└── 📁models
    └── __init__.py
    └── layers.py
    └── models.py
    └── utils.py
└── 📁utils
    └── __init__.py
    └── augmentation.py
    └── utils.py
└── check_io.py
└── LICENCE
└── preprocess.py
└── README.md
└── requirements.txt
└── train.py
```

## Usage
### Dataset preparation

You can drag and drop the folder containing your training audio files into the `/data/dataset/raw_sample/train/` folder and your test audio files into the `/data/dataset/raw_sample/test/` folder.

For IPT classes, test and train folders must share the same name. The class label is retrieved from the name of your IPT class folders.
```
└── 📁test
    └── 📁myTestDataset
        └── 📁IPTclass_1
            └── audiofile1.wav
            └── audiofile2.wav
            └── ...
        └── 📁IPTclass_2
            └── audiofile1.wav
            └── audiofile2.wav
            └── ...
        └── ...
└── 📁train
    └── 📁myTrainingDataset
        └── 📁IPTclass_1
            └── audiofile1.wav
            └── audiofile2.wav
            └── ...
        └── 📁IPTclass_2
            └── audiofile1.wav
            └── audiofile2.wav
            └── ...
        └── ...
```

You can use multiple training datasets. They must share the same names for IPT classes as well.

```
└── 📁train
    └── 📁myTrainingDataset1
        └── 📁IPTclass_1
        └── 📁IPTclass_2
        └── ...
    └── 📁myTrainingDataset2
        └── 📁IPTclass_1
        └── 📁IPTclass_2
        └── ...
    └── ...
```

### Preprocess your datasets
To preprocess your datasets, use the following command. The only required argument is `--name`.
```
python preprocess.py --name project_name
```

Other arguments:
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `--val_split`          | Specify which dataset the validation set will be generated.       | `train`, `test` | `train`            |
| `--val_ratio`          | Amount of validation samples.                                        | 0 <= Float value < 1  | `0.2`           |

A CSV file will be saved in the `/data/dataset/` folder with the following syntax:
```
project_name_dataset_split.csv
```

### Training
There are many different configurations for training your model. The only required argument is `--name`.
To train your model use the following command.
```
python train.py --name project_name
```
You can use the following arguments if you want to test different configurations.
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `--config`          | Name of the model's architecture.                                  | `v1`, `v2`, `one-residual`, `two-residual`, `transformer` | `v2`            |
| `--device`          | Specify the hardware on which computation should be performed.     | `cpu`, `cuda`, `mps`           | `cpu`           |
| `--gpu`             | Specify which GPU to use.                                          | `0`, `1`, ...                  | `0`             |
| `--sr`              | Sampling rate for downsampling the audio files.                    | `16000`, `22050`, `24000`, ... (Hz)| `24000`         |
| `--segment_overlap` | Overlap between audio segments.                                    | `True`, `False`                | `False`         |
| `--fmin`            | Minimum frequency for Mel filters.                                 | 0 < Float value (Hz) or `None` | `None`          |
| `--lr`              | Learning rate.                                  | 0 < Float value                | `0.001`         |
| `--epochs`          | Number of training epochs.                                         | 0 < Integer value              | `100`           |
| `--augment`         | Specify which augmentations to use.                                            | `pitchshift`, `timeshift`, `addnoise`, `polarityinversion`, `gain`, `hpf`, `lpf` or `all`| `pitchshift` |
| `--early_stopping`  | Number of epochs without improvement before early stopping.         | 0 < Integer value or `None`   | `None`          |
| `--reduceLR`        | Reduce learning rate if validation plateaus.                       | `True`, `False`                | `False`         |
| `--export_ts`       | Export the model as a TorchScript file (`.ts` format).              | `True`, `False`               | `False`         |
| `--save_logs`       | Save logs to disk.                                               | `True`, `False`               | `True`         |

Training your model will create a `runs` folder with the name of your project.
After training, the script automatically saves the model checkpoints in the `/runs/project_name/` folder.
If you use `--export_ts True`, the `.ts` file will be saved in the same folder.

```
└── 📁runs
    └── 📁project_name
```

`--save_logs True` will create a folder of the name of your project into the `logs` folder.
This folder will include a `results.csv` file containing the model's metrics as well as a CSV file for the confusion matrix.
```
└── 📁logs
    └── 📁project_name
```

### Running the model in real-time

To run your model in real time, you need first to check available audio input devices of your computer with the script `check_io.py`.

```
python check_io.py
```

This will display a list of the devices and their respective ID.

```
Input Device ID  0  -  MacBook Pro Microphone
Input Device ID  1  -  BlackHole 2c
Input Device ID  2  -  BlackHole 16ch
```

Once you have found your device ID, use the command `python realtime.py` to start using your model in real time. Arguments `--name`, `--input` and `--channel`are required.
The script will automatically take the most recent model of your project.
```
python realtime.py --name your_project --input 0 --channel 1 
```

Other arguments:
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `--device`          | Specify the hardware on which computation should be performed.      | `cpu`, `cuda`, `mps`           | `cpu`           |
| `--gpu`             | Specify which GPU to use.                                           | `0`, `1`, ...                  | `0`             |
| `--buffer_size`     | Specify audio buffer size.                                       | Integer                        | `256`           |
| `--moving_average`  | Window size for smoothing predictions with a moving average.        | Integer                        | `5`             |

Predictions [0, n_class-1] are sent via UDP through port 5005 with a `/class` adress.
Use an UDP receiver to retrieve the predictions.

## Related works
If you use this code in your research, please cite the following papers.

```
@inproceedings{brochec:hal-04642673,
  TITLE = {{Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques}},
  AUTHOR = {Brochec, Nicolas and Tanaka, Tsubasa and Howie, Will},
  URL = {https://hal.science/hal-04642673},
  BOOKTITLE = {{International Computer Music Conference (ICMC 2024)}},
  ADDRESS = {Seoul, South Korea},
  YEAR = {2024},
  MONTH = Jul,
  PDF = {https://hal.science/hal-04642673/file/Brochec_Microphone_based_Data_Augmentation_for_Automatic_Recognition_of_Instrument_Playing_Techniques_.pdf},
  HAL_ID = {hal-04642673},
  HAL_VERSION = {v1},
}

@inproceedings{fiorini:hal-04635907,
  TITLE = {{Guiding Co-Creative Musical Agents through Real-Time Flute Instrumental Playing Technique Recognition}},
  AUTHOR = {Fiorini, Marco and Brochec, Nicolas},
  URL = {https://hal.science/hal-04635907},
  BOOKTITLE = {{Sound and Music Computing Conference (SMC 2024)}},
  ADDRESS = {Porto, Portugal},
  YEAR = {2024},
  MONTH = Jul,
  KEYWORDS = {AI ; Co-creativity ; Instrumental playing techniques ; Multi-agent system ; Somax2},
  PDF = {https://hal.science/hal-04635907/file/SMC2024_GUIDING_CO_CREATIVE_MUSICAL_AGENTS_THROUGH_REAL_TIME_FLUTE_INSTRUMENTAL_PLAYING_TECHNIQUE_RECOGNITION_CAMERA_READY.pdf},
  HAL_ID = {hal-04635907},
  HAL_VERSION = {v1},
}
```
### Other related paper
• Nicolas Brochec and Tsubasa Tanaka. [Toward Real-Time Recognition of Instrumental Playing Techniques for Mixed Music: A Preliminary Analysis](https://hal.science/hal-04263718). International Computer Music Conference (ICMC 2023), Oct 2023, Shenzhen, China.

### Datasets
• Nicolas Brochec and Will Howie. [GFDatabase: A Database of Flute Playing Techniques](https://doi.org/10.5281/zenodo.10932398) (version 1.0.1). Zenodo, 2024.

## Acknowledgments
This project uses code from the [pytorch_balanced_sampler](https://github.com/khornlund/pytorch-balanced-sampler) repository created by Karl Hornlund.

## Funding
This work has been supported by the [ERC Reach](https://reach.ircam.fr/) (Raising Co-creativity in Cyber-Human Musicianship), hosted at [IRCAM](https://www.ircam.fr/recherche/equipes-recherche/repmus/), directed by Gérard Assayag.
