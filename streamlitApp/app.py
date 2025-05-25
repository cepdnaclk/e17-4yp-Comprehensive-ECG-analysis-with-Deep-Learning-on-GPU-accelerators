import streamlit as st
import wfdb
import pandas as pd
import os
import numpy as np
import torch
import plotly.graph_objects as go
import matplotlib.pyplot as plt

from KanResWide_X2 import KanResWide_X2

# df contains the IDs to map ECG files
# ecg_id is the column name in the dataframe
df = pd.read_csv("/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/PTBXLpreprocessed.csv")


# Set the app title
st.title("ECG Parameter Variations")

# List of ECG parameters for prediction
parameter_options = ["HR", "QT", "QRS", "PR"]

# Dropdown menu for selecting parameter
selected_parameter = st.selectbox("Select Parameter for Prediction", parameter_options)

#loading the validated models according to the selected parameter
input_shape = (8, 5000)  
# Number of output units
output_size = 1 
model = KanResWide_X2(input_shape, output_size)

# Proceed with further steps based on the selected parameter
if selected_parameter:
  # Use multiple if/elif statements for specific conditions
  if selected_parameter == "HR":
    checkpointPath = '/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints/PTBXL/hr_best_.pt'
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint)
    st.success("HR model loaded successfully.")
  elif selected_parameter == "QT":
    checkpointPath = '/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints/PTBXL/qt_best_.pt'
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint)
    st.success("QT model loaded successfully.")
  elif selected_parameter == "QRS":
    checkpointPath = '/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints/PTBXL/qrs_best_.pt'
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint)
    st.success("qrs model loaded successfully.")
  elif selected_parameter== "PR":  # Handles "PR" or any other option
    checkpointPath = '/storage/localSSD/e17-4yp-comprehensive-ecg-analysis/e17-4yp-Comprehensive-ECG-analysis-with-Deep-Learning-on-GPU-accelerators/python-scripts-resnet/checkpoints/PTBXL/pr_best_.pt'
    checkpoint = torch.load(checkpointPath)
    model.load_state_dict(checkpoint)
    st.success("PR model loaded successfully.")
else:
  st.write("Please select a parameter for prediction.")

# enter ECG IDs
# Text input for ECG IDs
ecg_ids_text = st.text_input("Enter comma-separated ECG IDs (e.g., 123,456,789)", key="ecg_ids")

# nparray to hold the predictions
predictions = []

# Proceed if IDs are entered
if ecg_ids_text:
    # Split the entered text by comma to get a list of IDs
    ecg_ids = ecg_ids_text.split(",")
    # Display the entered IDs
    st.write("Entered ECG IDs:", ecg_ids)
    # turn the ecg_ids into integers
    ecg_ids = [int(x) for x in ecg_ids]

    # for loop to iterate through the IDs
    for ecg_id in ecg_ids:
        # Check if the entered ID is in the dataframe
        if ecg_id in df.ecg_id.values:
            st.write(f"ECG ID {ecg_id} is valid.")
            # Get the filename from the row
            # file path
            file_index = int(ecg_id)
            folder_name = str(file_index // 1000).zfill(2)+'000' 
            file_name = str(file_index).zfill(5)+'_hr'

            ecg_record_path = os.path.join(os.path.dirname(os.getcwd())  ,  'data/data', 'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1', 'records500', folder_name, file_name)

            st.write(f"Filename: {ecg_record_path}") 

            # Use wfdb.rdsamp to read both the .dat file and .hea header file
            ecg_record_data, ecg_record_header = wfdb.rdsamp(ecg_record_path)

            # ['I', 'II', 'III', 'AVR', 'AVL', 'AVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] - 12 original channels
            # ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'] - 8 channels as deepfake data set

            columns_to_remove = [2, 3, 4, 5]    # remove channels - 'III', 'AVR', 'AVL', 'AVF'
            ecg_record_data = np.delete(ecg_record_data, columns_to_remove, axis=1)

            ecg_signals = torch.tensor(ecg_record_data) # convert dataframe values to tensor
            #print(ecg_signals.max())
            #print("*********************")
            ecg_signals = ecg_signals.float()
        
            # Transposing the ecg signals
            #print(ecg_signals.shape)
            #exit()
            ecg_signals = ecg_signals/6   # normalization
            #print(ecg_signals)
            #exit()
            ecg_signals = ecg_signals.t() 

            # predicting the desired parameter
            # Move data to GPU if available
            if torch.cuda.is_available():
                ecg_signals = ecg_signals.to('cuda')
                model.to('cuda')

            # Set model to evaluation mode (optional)
            model.eval()

            # Make prediction
            output = model(ecg_signals.unsqueeze(0))  # Add batch dimension (1, 8, 5000)

            # Interpret output (assuming output is a single value)
            prediction = output.item()  # Extract the predicted value
            predictions.append(prediction)
            st.write(f"Predicted {selected_parameter}: {prediction:.2f}")  # Print with 2 decimal places

        else:
            st.warning(f"ECG ID {ecg_id} is invalid.")

else:
    st.warning("Please enter ECG IDs separated by commas. DON'T include spaces between the commas.")

# Plot the predictions by matplotlib
# Generate the plot using Matplotlib
'''
plt.figure(figsize=(8, 5))  # Set figure size (optional)
plt.plot(predictions)
plt.xlabel("Index")
plt.ylabel(f"Prediction Value of {selected_parameter}")
plt.title("Line Graph of Predictions")

# Display the plot in Streamlit
st.pyplot()
'''

#st.write(predictions)
# ecg_ids = 1, 21807, 21808,21809,21810,2,3,4,5,6,7
test_pr = [154,132,158,180,124,152,134,146,148,156,140]
test_qt = [404,378,422,416,386,412,430,368,396,374,432]

# Plot the predictions by plotly

timestamps = list(range(len(predictions)))  # List of indexes representing timestamps

# Configure the Plotly graph
#fig = go.Figure(data=go.Scatter(x=timestamps, y=predictions, mode='lines'))


fig = go.Figure()
fig.add_trace(go.Scatter(x=timestamps, y=predictions, mode='lines', name='Model Predictions',line=dict(color='lightblue')))
fig.add_trace(go.Scatter(x=timestamps, y=test_pr, mode='lines', name='Test Values',line=dict(color='orange')))


fig.update_layout(
    title=f"Plot of {selected_parameter} Predictions and actual values",
    xaxis_title="Time",  # Adjust title based on your x-axis data
    yaxis_title=f"{selected_parameter} Value (ms)",  
)

# Display the graph in Streamlit
st.plotly_chart(fig)



# Upload ECG data (multiple files allowed)
# make it compatible with ptbxl file type as well
uploaded_files = st.file_uploader("Upload ECG files", type=['asc'], accept_multiple_files=True)

