# Azimuth-Prediction
Automated antenna azimuth prediction project developed in collaboration with Dialog Axiata PLC. Detects correct azimuth based on input data.
# Antenna Azimuth Prediction

This project aims to predict the correct azimuth for antennas based on input features such as Cell Key, RSRP (Reference Signal Received Power), Instance Count, and Cell Azimuth.

## Input Features

- **Cell Key**: A unique identifier for each antenna.
- **RSRP (Reference Signal Received Power)**: The average signal power received by the relevant antenna from the given coordinate.
- **Instance Count**: The number of signals received by the antenna from the given coordinate.
- **Cell Azimuth**: Labeled azimuth for the antenna.
- **Longitude** and **Latittude**: Coordinates of the location the signal co ing from.

## Usage

1. **Data Preparation**: Ensure your dataset includes the required input features.
2. **Training the Model**: Trained 3 models. Random Forest Regression, Linear Regression and ANN.
3. **Prediction**: Utilize the trained model to predict the azimuth for new antenna data.

## Directory Structure

- `src/`: Includes source code for data preprocessing, model training, and prediction.

## Contributors

- Thisura Thibbotuge : https://github.com/thisura01
- Chathuni Kandawinna : https://github.com/ChathuniiK
- Raveesha Dulmi
- Ravisha Nilneth
- Sashani Liyanage

## Collaboration

This project was conducted in collaboration with [Dialog Axiata PLC](https://www.dialog.lk/). For inquiries or contributions, please contact thisurathibbotuge@gmail.com.
