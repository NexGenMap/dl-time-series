#!/bin/bash

wget "https://storage.googleapis.com/nextgenmap-dataset/dl-time-series/deforestation_toy.zip"

rm -vR deforestation_toy
unzip deforestation_toy.zip

echo "Executing generate_img_series.py"
./generate_img_series.py -i deforestation_toy/images -b 1 2 3 4 -o deforestation_toy/image_series

echo "Executing generate_ts.py"
./generate_series.py -i deforestation_toy/image_series/ -s deforestation_toy/samples/toy_samples.shp -o deforestation_toy/data_series/ -n 2

echo "Executing train_model.py"
./train_model.py -i deforestation_toy/data_series/ -e 100 -o deforestation_toy/model

echo "Executing classify_img_series.py"
./classify_img_series.py -i deforestation_toy/image_series/ -m deforestation_toy/model/last_model.H5 -o deforestation_toy/classification/result.img