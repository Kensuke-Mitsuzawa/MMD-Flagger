# 🔧 Setup Instructions

## 🐍 Python Environment

This project uses Python 3.9.12.

## 📦 Dependency Installation
We use poetry for dependency management. Install it and set up the environment:

```
pip install poetry
poetry install
```

Poetry will automatically create and manage a virtual environment.

## 🛠 Troubleshooting

For fairseq compilation errors:

```
sudo apt install build-essential
```

For compatibility issues with fairseq, try:

```
pip install pip==24.0
```

# 📁 LFAN-HALL Model Files

Translating LFAN-HALL dataset requires the model files provided by [the authors](https://github.com/deep-spin/hallucinations-in-nmt).

The model files are downloadable via the following links (confirmed at 2025-05-18).

https://www.mediafire.com/file/mp5oim9hqgcy8fb/checkpoint_best.tar.xz/file
https://www.mediafire.com/file/jfl7y6yu7jqwwhv/wmt18_de-en.tar.xz/file



```
mkdir model_guerreiro_2023
cd model_guerreiro_2023
wget [URL checkpoint_best.tar.xz]
wget [URL wmt18_de-en.tar.xz]
tar -xvf checkpoint_best.tar.xz
tar -xvf wmt18_de-en.tar.xz
git clone git@github.com/deep-spin/hallucinations-in-nmt
mv hallucinations-in-nmt/sentencepiece_models .
```

# ⚙️ TNG Implementation (Raunak et al., 2021)

Our code uses the TNG module from the NL-Augmenter repository.
This requires Python 3.7 (tested on 3.7.17).

## 🐍 Python 3.7 Installation
Using pyenv is recommended.
To build manually:

```
wget https://www.python.org/ftp/python/3.7.17/Python-3.7.17.tar.xz
tar -xf Python-3.7.17.tar.xz
cd Python-3.7.17
./configure --prefix="$HOME/.local" --enable-optimizations
make -j$(nproc)
make install
```

## 🧱 NL-Augmenter Installation

```
git clone https://github.com/GEM-benchmark/NL-Augmenter
cd NL-Augmenter
python3.7 setup.py sdist
pip3.7 install -e .
pip3.7 install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.0.0/en_core_web_sm-3.0.0.tar.gz
```

# 🧪 Testing

Place the downloaded model files in:

```
tests/testresources/model_guerreiro_2023
```

Run tests with:

```
pytest tests
```

# 📊 Datasets

Two necessary dataset files are found at `tests/testresources/eval_datasets`.

The `LFAN-HALL` dataset is originally from [Github Page](https://github.com/deep-spin/hallucinations-in-nmt/tree/main/data), but modified by us.

The `Halomi` dataset is from [Github Page](https://github.com/facebookresearch/stopes/tree/main/demo/halomi).


# 🔁 Reproduction Instructions

Update the corresponding .toml config files before running the scripts below.
The base config files are at `./config_files`.


## 🧪 Natural Competitor Baselines

### LFAN-HALL

```
python assessment/ver1/lfan_hall/flagging_interface.py -c <config-path> -m flag
python assessment/ver1/lfan_hall/flagging_interface.py -c <config-path> -m eval
```

### Halomi

```
python assessment/ver1/halomi/flagging_interface.py -c <config-path> -m flag
python assessment/ver1/halomi/flagging_interface.py -c <config-path> -m eval
```

## 🚩 MMD-Flagger (Our Method)

```
python assessment/ver3/flagging_interface.py -c <config-path> -m translation
python assessment/ver3/flagging_interface.py -c <config-path> -m flag
python assessment/ver3/flagging_interface.py -c <config-path> -m eval
```

## 📈 Appendix: Stability Evaluation

```
python assessment/appendix/MMD-Flagger_stability/check_mmd_flagger_stability.py -p <config-path> -m translation
python assessment/appendix/MMD-Flagger_stability/check_mmd_flagger_stability.py -p <config-path> -m flag
```


