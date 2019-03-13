#!/bin/bash
rm -vR deforestation_toy

unzip deforestation_toy.zip

echo "Executing tack_imgs.py"
./stack_imgs.py -i deforestation_toy/images -b 1 2 3 4 -o deforestation_toy/image_series

echo "Executing generate_ts.py"
./generate_series.py -i deforestation_toy/image_series/ -s deforestation_toy/samples/toy_samples.shp -o deforestation_toy/data_series/ -n 2

echo "Executing train_model.py"
./train_model.py -i deforestation_toy/data_series/ -e 100 -o deforestation_toy/model

echo "Executing classify_imgs.py"
./classify_imgs.py -i deforestation_toy/image_series/ -m deforestation_toy/model/last_model.H5 -o deforestation_toy/classification/result.img