# Real-Time Instrumental Playing Techniques Recognition

![Status: Not Ready](https://img.shields.io/badge/status-not%20ready-red)

## Installation
Clone this repository.
```
git clone https://github.com/nbrochec/realtimeIPTrecognition/
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
To preprocess your datasets, use the following command. he only required argument is `--name`.
```
python preprocess.py --name project_name
```

Other arguments:
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `--val_split`          | Specify on which dataset the validation set will be generated.       | `train` or `test` | `train`            |
| `--val_ratio`          | Amount of validation samples.                                        | Float value < 1  | `0.2`           |

A CSV file will be saved in the `/data/dataset/` folder with the following syntax:
```
project_name_dataset_split.csv
```

### Training
There are many different configurations for training your model. The only required argument is `--name`.
```
python train.py --name project_name
```
You can use the following arguments if you want to test different configurations.
| Argument            | Description                                                         | Possible Values                | Default Value   |
|---------------------|---------------------------------------------------------------------|--------------------------------|-----------------|
| `--config`          | Name of the model's architecture.                                  | `v1`, `v2`, `one-residual`, `two-residual`, `transformer` | `v2`            |
| `--device`          | Device to use for training.                                        | `cpu`, `cuda`, `mps`           | `cpu`           |
| `--gpu`             | GPU selection to use.                                              | `0`, `1`, ...                  | `0`             |
| `--sr`              | Sampling rate to downsample the audio files.                        | 16000, 22050, 24000           | `24000`         |
| `--segment_overlap` | Overlap between audio segments.                                    | `True`, `False`                | `False`         |
| `--fmin`            | Minimum frequency for Mel filters.                                 | Numerical value (Hz)           | `None`          |
| `--lr`              | Learning rate for the optimizer.                                   | 0.001, 0.01, etc.              | `0.001`         |
| `--epochs`          | Number of training epochs.                                         | Integer value                  | `100`            |
| `--early_stopping`  | Number of epochs without improvement before early stopping.         | Integer value or `None`        | `None`          |
| `--reduceLR`        | Reduce learning rate if validation plateaus.                       | `True`, `False`                | `False`         |
| `--export_ts`       | Export the model as a TorchScript file (`.ts` format).              | `True`, `False`                | `False`         |

Training your model will create a `runs` folder with the name of your project.
After training, the script automatically saves the best model checkpoints in the `/runs/project_name/` folder.
If you use `--export_ts True`, the `.ts` file will be saved in the same folder.

```
└── 📁runs
    └── 📁project_name
```

### Running the model in real-time

[...]

## Related research works
• Nicolas Brochec, Tsubasa Tanaka, Will Howie. [Microphone-based Data Augmentation for Automatic Recognition of Instrumental Playing Techniques](https://hal.science/hal-04642673). International Computer Music Conference (ICMC 2024), Jul 2024, Seoul, South Korea.

• Nicolas Brochec and Tsubasa Tanaka. [Toward Real-Time Recognition of Instrumental Playing Techniques for Mixed Music: A Preliminary Analysis](https://hal.science/hal-04263718). International Computer Music Conference (ICMC 2023), Oct 2023, Shenzhen, China.

• Marco Fiorini and Nicolas Brochec. [Guiding Co-Creative Musical Agents through Real-Time Flute Instrumental Playing Technique Recognition](https://hal.science/hal-04635907). Sound and Music Computing Conference (SMC 2024), Jul 2024, Porto, Portugal.

## Related Datasets
• Nicolas Brochec and Will Howie. [GFDatabase: A Database of Flute Playing Techniques](https://doi.org/10.5281/zenodo.10932398) (version 1.0.1). Zenodo, 2024.

## Acknowledgments

This project uses code from the [pytorch_balanced_sampler](https://github.com/khornlund/pytorch-balanced-sampler) repository created by Karl Hornlund.

## Funding

This work has been supported by the ERC Reach (Raising Co-creativity in Cyber-Human Musicianship) directed by Gérard Assayag.
