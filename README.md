# Cropify - Crop Recommendation System

An application that suggests a suitable crop based on the critical params (NPK levels, soil moisture, temperature, humidity, pH).

## Setup
1. Clone the repo and create a virtual environment using

   ```python
   python -m venv run
   ```

2. Activate the virtual environment using

   ```python
   .\run\Scripts\activate
   ```

3. Install the requirements using

   ```python
   pip install -r requirements.txt
   ```

4. Start the server using

   ```python
   python manage.py runserver
   ```

5. Open localhost:8000 in the browser to launch the application.

## Usage

1. Click on register and provide the necessary inputs.
2. Fill in the values for N,P,K levels and pH.
3. Click on submit and you'll be shown the suggested crop (temperature, humidity and soil moisture will be taken from the sensors accordingly).

## Additional steps to own the port if you're using Linux

1. Switch into superuser mode using

   ```shell
   sudo su
   ```

2. Navigate to dev directory in root using

   ```shell
   cd /dev
   ```

3. Change the port ownership using

   ```shell
   chown username ttyUSB0
   ```

## Note

This application was last tested successfully with Python 3.11.3.

The dataset used to train the model is present in the repo with the name crop_recommendation.csv.

The trained model is also present in the repo with the name finalized_model.sav. The code used to train the model is present in model.py (this is very raw, we've tried out multiple models and chose RandomForestClassifier).

The Arduino code is present under dht folder.

You mean need to change the port that's being used with Arduino, this can be changed in line 14 in views.py. If you're on Windows, 'com3' or a similar one would be the preffered choice.