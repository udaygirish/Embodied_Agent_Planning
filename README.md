# Language Embodied Navigation using Local and Global Planners

This project aims to convert human language-defined tasks into a sequence of actions understandable by a Robot. The inspiration comes from the Facebook AI Habitat Challenge.

## Project Dependencies

Please follow the steps below to set up the required dependencies for the project:

1. **[Habitat-sim](https://github.com/facebookresearch/habitat-sim)**
2. **[Habitat-lab](https://github.com/facebookresearch/habitat-lab)**

### Other Dependencies

Make sure to install the following dependencies:

1. **Speech Recognition Module**
    ```bash
    pip install SpeechRecognition==3.10.1
    ```

2. **PortAudio19-dev**
    ```bash
    sudo apt-get install portaudio19-dev
    ```

3. **PyAudio**
    ```bash
    pip install pyaudio==0.2.14
    ```

4. **NLTK (Natural Language Toolkit)**
    ```bash
    pip install nltk
    ```

5. **TensorFlow**
    ```bash
    pip install tensorflow
    ```

6. **Scikit-learn**
    ```bash
    pip install scikit-learn
    ```

## Project Overview

Language is something we humans use to connect
with each other to express emotions/opinions, to delegate tasks.
It’s almost impossible for a human to give another a JSON file to
execute something, so language is the primary aspect of everyday
life. Our project inspiration started with how we can convert
a task defined in Human language can be converted to a set
of sequences of actions that can be understood by a Robot.
The project is inspired by the FacebookAI Habitat Challenge
Object Nav theme. The
objective is to develop agents that can navigate unfamiliar
environments and move away from closed object classes towards
open-vocabulary natural language. So the challenge is for a robot
(virtual/real) when placed in an unknown environment should be
able to navigate the environment by using information from its
State space (From IMU + GPS/Encoders) and Image (visual cues)
information.


## Usage
1. Download the [Project Voice2Nav](https://drive.google.com/file/d/1PuzDmZllEMJbZ_cPGTRzxghBE4v2ZLUi/view?usp=drive_link) and extract it into habitat-sim folder
2. Run the python script
   Scene no : 1-6
   ```bash
   python main.py -s <SCENE_NO>

3. Note that we have provided some sample scenes we used from the Matterport HM3D Data after getting access from them. For more scenes please fill out their application for Access and download and use. The example scenes we have given in this repo are purely for ACADEMIC USE.
4. For more understanding please email us.
