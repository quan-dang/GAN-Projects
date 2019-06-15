# Face Aging using cGANs

## Dataset

This project use the [dataset](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/static/wiki_crop.tar) from the authors of the paper
titled _Deep expectation of real and apparent age from a single image without facial landmarks_.

## Project structure

* __wiki_crop/__
    - __00/__
    - __01/__
    - __...__
* __logs/__
* __results/__
* __utils.py__
* __main.py__

## How to run

1. __Step 1__: Preparation

Download the dataset to the root path of the project.

2. __Step 2__: Run cGANs

```
python main.py
```
