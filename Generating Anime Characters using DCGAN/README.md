# Generating Anime Characters using DCGAN

## Project structure
* __gallery-dl/__: our datasets
    - __danbooru/__:
        - __face/__: original crawled images
            - ..
            - ..
        - __face-cropped/__: cropped crawled images
            - ..
            - ..
* __logs/__: for tensorboard visualization
* __utils.py__: utility functions
* __preprocessing.py__: preprocess crawled images
* __main.py__: main function

## How to run

1. __Step 1__: Download the dataset

Go to the root path of our project and install _gallery-dl_

```
pip install --upgrade gallery-dl
```

Crawl anime images (about 10,000 images) by executing the following command:

```
gallery-dl https://danbooru.donmai.us/posts?tags=face
```

2. __Step 2__: Crop and resize images in the dataset

Install _python-animeface_ 

```
pip install animeface
```

Crop and resize images executing the following command:

```
python preprocessing.py
```

3. __Step 3__: Execute our main script

```
python main.py
```