![Computer Vision](https://img.shields.io/badge/Computer%20Vision-blue) 
![Operation Research](https://img.shields.io/badge/Operation%20Research-grey) 
![LLM](https://img.shields.io/badge/LLM-white) 
![Retrieval Augmented Generation](https://img.shields.io/badge/Retrieval%20Augmented%20Generation-orange) 
![NVIDIA NIM](https://img.shields.io/badge/NVIDIA%20NIM-green) 
![LangChain](https://img.shields.io/badge/LangChain-white) 
![YOLOV8n](https://img.shields.io/badge/YOLOv8n-purple) 
![FAISS](https://img.shields.io/badge/FAISS-pink) 
![Opencv](https://img.shields.io/badge/OpenCV-red) 
![Streamlit](https://img.shields.io/badge/Streamlit-grey) 



# Vision-Guided RAG Agent for Risk Prevention and Root Cause Analysis with NVIDIA and LangChain

## Table of Contents
1. [Introduction](#introduction)
    - [Overall Project Architecture](#overall-project-architecture)
    - [DEMO](#demo)
3. [Setup Instructions](#setup-instructions)
    - [Clone Repository](#clone-repository)
    - [Open in VS Code](#open-in-vs-code)
    - [Create and Activate Virtual Environment](#create-and-activate-virtual-environment)
    - [Install Dependencies](#install-dependencies)
4. [First Part: Computer Vision](#first-part-computer-vision)
    - [Train the YOLOv8 Model](#train-the-yolov8-model)
    - [Inference with trained YOLOv8 Model(optional)](#inference-with-yolov8-model)
5. [Second Part: PFMEA (Operation Research)](#second-part-pfmea-operation-research)
    - [Process Failure Mode Effect and Analysis](#process-failure-mode-effect-and-analysis)
6. [Third Part: LLM using NVIDIA NIM](#third-part-llm-using-nvidia-nim)
    - [NVIDIA NIM APIs or Endpoints](#nvidia_nim)
    - [Run LLM Response](#run-llm-response)
7. [Contact](#contact)

# Introduction
This project unlocks the combined potential of Computer Vision, Operational Research, and Large Language Models (LLMs) using Retrieval Augmented Generation (RAG) to determine the root cause and eliminate the high risk involved in processes by taking preventive action. 

This project has been developed for the NVIDIA Generative AI Developer Contest. It consists of three main parts: 
- **Computer Vision** using YOLOv8
- **PFMEA** (Process Failure Mode Effect and Analysis)
- **LLM** (Large Language Model) Llama-3 using NVIDIA NIM

This intelligent agent detects plant status, analyzes feedback, stores it, identifies potential and root causes, and effectively responds to user queries with a knowledge base.

## Overall Project Architecture
<img src="https://github.com/Dherya27/AI-Driven-Agent-for-Risk-Prevention-and-Root-Cause-Analysis-using-NVIDIA-NIM/blob/main/img/overall_architecture.png" width=675 heigh=425>

## Demo
<img src="https://github.com/Dherya27/AI-Driven-Agent-for-Risk-Prevention-and-Root-Cause-Analysis-using-NVIDIA-NIM/blob/main/img/ai-agent-ezgif.com-video-to-gif-converter.gif" width=675 heigh=425>


# Setup Instructions
## Clone Repository
```bash
$ git clone https://github.com/Dherya27/check.git
```
## Open in VS code
```bash
$ cd <project-folder>
$ code .
```

## Create and Activate Virtual Environment
```bash
$ conda create -n nvidia_contest python=3.10
$ conda activate nvidia_contest
```
## Install Dependencies
```bash
$ pip install -r requirements.txt
```

# First Part: Computer Vision
## Train the YOLOv8 Model
Before training the model, ensure your data is organized correctly:

"data" folder should contain train and valid folders.
Each folder should have "images" and "labels" subfolders.

Create a data.yaml file as shown in the code.

To train the model, run the following command:
```bash
$ python yolo_train.py
```
## Inference with trained YOLOv8 Model(optional)
To perform inference:

Uncomment the last two lines in yolo_inference.py script, and then
run the command:
```bash
$ python yolo_inference.py
```
This will display predicted classes in your terminal.

# Second Part: PFMEA (Operation Research)
## Process Failure Mode Effect and Analysis
### Process Failure Mode Effect and Analysis(PFMEA) incorporates:

- Cause and Effect Analysis (Isikawa/Fishbone Diagram), and 
- Pareto Principle (80-20 principle) to identify potential causes.
- Calculation of Risk Priority Number (RPN) using Severity, Occurrence, and Detectability, where

   Risk Priority Number(RPN) = (Severity) x (Coccurance) x (Detectability)

High RPN (Risk Priority Number) values indicate critical causes(root cause) that should be prioritized for elimination to avoid high risks.
By focusing on high RPN values, you can identify and address the most critical issues in the process to enhance safety and efficiency.

# Third Part: LLM using NVIDIA NIM

## NVIDIA NIM APIs or Endpoints
Get the foundational models from the NVIDIA NIM APIs or endpoints:
- [NVIDIA NIM Explore](https://build.nvidia.com/explore/discover)
- [NVIDIA AI Endpoints Documentation](https://python.langchain.com/v0.2/docs/integrations/chat/nvidia_ai_endpoints/)

### Model and API Keys
Go to the above links, login, and click on any foundational model. Click on "Get API Key" to obtain your API key as shown in the image below:

<img src="https://github.com/Dherya27/AI-Driven-Agent-for-Risk-Prevention-and-Root-Cause-Analysis-using-NVIDIA-NIM/blob/main/img/api_key_sample.png" width=300 height=200>

- Create a `.env` file in the root directory and save your API key:
  ```env
  NVIDIA_API_KEY="-------your_API_Key-----------"
  ```


## Run LLM Response
To get the LLM response, run the following command:
```bash
$ streamlit run main.py
```

# Contact
If you encounter any issues or have suggestions for industrial applications, please feel free to contact me:
- LinkedIn: https://www.linkedin.com/in/dherya27/
- Email: dheryarobotics27@gmail.com

Your feedback and collaboration are greatly appreciated.





