# Installation
 
The following installation steps were tested with Debian 9 (stretch) and Ubuntu 18.04 (Bionic Beaver). Please, run all commands either as root or sudoer user.

### 1) To install python3 and some dependencies, one must run, in terminal:

`apt update && apt install -y python3 python3-gdal python3-pip python3-dev wget`
 
### 2) Install TensorFlow, Keras and scikit-learn:
`pip3 install tensorflow keras sklearn`

### 3) Install RIOS:
`wget https://bitbucket.org/chchrsc/rios/downloads/rios-1.4.5.tar.gz && tar -xvzf rios-1.4.5.tar.gz && cd rios-1.4.5 && python3 setup.py install --prefix=/opt/rios-1.4.5 && export PATH=$PATH:/opt/rios-1.4.5/bin/ && export PYTHONPATH=/opt/rios-1.4.5/lib/python$(python3 --version | cut -c8-10)/site-packages/ && cd .. && rm -rf rios-1.4.5 rios-1.4.5.tar.gz
`
# Usage Example

Default data directory is **./data**, but you can place a custom directory in the optional [data_dir].

### Training the model

`python3 run.py train [data_dir]`

### Evaluating the model

`python3 run.py eval [data_dir]`

### Running predict over the entire scene

`python3 run.py predict [data_dir]`

****Important: the current commit's default data directory doesn't provide example data (images). In order to run throughout all steps, user must provide its own directory for now.****
