# NVIDIA GENAI Contest Project






















# Step to run this code:

Go to your local terminal, and run command:
$ git clone "............................"

Open this folder in VScode

# Create a vitual environment, by running command:
$ conda create -n nvidia_contest python=3.10

# Activate the virtual environmemt
$ conda activate nvidia_contest

# Install all libraries and dependencies
$ pip install -r requirements.txt

################################ First Part : Computer Vision #####################################
### Once you have successfully installed, Then train the YOLOv8 model:
# To train the model, you need to prepare data, keep the images in similar way for train and valid folder, inside each folder, there should be be images and labels folders.

# create a data.yaml file as shown in the code 

# Train the YOLOv8 model: 
python yolo_train.py

## Once trained, you can verify whether you vision model is predicting fine, by running yolo_inference.py file:
# Uncomment the last two line of this script, and run command:
python yolo_inference.py

# It should show the predicted class in your terminal

######################### SECOND PART: PFMEA(Operation Research) ####################################
PFMEA : Process Failure Mode Effect and Analysis
It majorly includes "Cause and Effect analysis"/"Isikawa diagram"/"Fishbone Diagram", "Pareto Principle"/"80-20 principle" to define relevant potential causes for an effect. Ultimately, we calculate "Risk Pripority Number(RPN)" = (Severity) x (Occurance) x (Detectability)

To estimate the values of (Severity), (Occurance), (Detectability), It requires the expert of this field.
Severity :
Occurance :
Detectability :

Once you set the values of these three for each potential causes, You are set to get the root cause based on high RPN values.

For this project, I have set these values based on articles, papers etc over the internet.

######################### THIRD PART: LLM using NVIDIA NIM ############################################

# To get the LLM response, run command:
streamlit run main.py


If any find any problems, Anyone who feels that this can be implemented into your industrial work, I would love to make something fruition together. Feel free to contact me on :
linkedIn: 
email: dheryarobotics27@gmail.com






