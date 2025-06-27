# Endless Runner with AI Controller

This repo is for Serious Game Endless Runner project

# Setup
1. Python `version=3.9`
1. Create conda environment
```cmd
conda create -n py39-env python=3.9
conda activate py39-env
```
1. Install necessary libraries with:
```cmd
pip install -r requirments.txt
```

# Dataset
1. Download dataset from [here](https://drive.google.com/drive/folders/1Eoz_s6n4jPc5ikg9J29R7dAT7iPIxese?usp=sharing)
1. Create a directory with name `data`
1. Extract the compressed dataset into `data` directory 

# Training Gesture
1. Open terminal and change directory to Gesture `cd Gesture`
1. Run `python main.py`

## Struktur dataset voice
```lua
--data
   |
   |--english--|--down
               |--left
               |--right
               |--up

```

## Hasil Voice Classification
### Tgl 26 Juni 2025 13:41
| No | Model        | Test Acc (%) |
|----|--------------|--------------|
| 1  | MLPClassifier | 93.54        |
| 2  | CNN1D         | 96.77        |

## Run inference model CNN 1D
1. Download model dari [sini](https://drive.google.com/file/d/1Irc6I4x7QZt6Xmbhfe3r2YchhOvp_up-/view?usp=sharing)
2. Ubah `MODEL_PATH` di file `inference_gui_cnn1.py` sesuai dengan path model pada poin 1.
3. Jalankan perintah `python inference_gui_cnn1.py`

# Setup C# TCP Server (GTK + Voice Visual Feedback)
## Requirements
1. .NET SDK 9.0
1. GTK# (GtkSharp for Linux/Mac/Windows)

## Install GtkSharp
```xml
<PackageReference Include="GtkSharp" Version="3.24.24.4" />
```

## Compile dan Run
```bash
dotnet run
```

## Build
```bash
dotnet build
```

## Publish for Linux
```bash
dotnet publish -c Release -r linux-x64 --self-contained true
```

## Publish for Windows
```bash
dotnet publish -c Release -r win-x64 --self-contained true
```