# svRepresentations

## 1. Environment setup

#### 1.1 Create and activate a new virtual environment

```
conda create -n dnabert python=3.6
conda activate dnabert
```



#### 1.2 Install the package and other requirements

(Required)

```
conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

git clone https://github.com/jerryji1993/DNABERT
cd DNABERT
python3 -m pip install --editable .
cd examples
python3 -m pip install -r requirements.txt
```

## 2. Download model

#### 2.1 Download pre-trained DNABERT

[DNABERT6](https://drive.google.com/file/d/1BJjqb5Dl2lNMg2warsFQ0-Xvn1xxfFXC/view?usp=sharing)

Download the pre-trained model in to a directory. (If you would like to replicate the following examples, please download DNABERT 6). Then unzip the package by running:

```
unzip 6-new-12w-0.zip
```

Put the unzipped file in a folder called dnabert6

## 3. Dataset Info

In the dataset folder two types of files will be found.

- csv files containing the metadata realtive to the SVs applied over a certain sequence (e.g sv type, breakpoints...)
- tsv files containing sequences already preprocessed (e.g divided in kmer) with the label of the SV applied over them