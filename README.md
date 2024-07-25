# Audio Classification

## Data 

## Data Collection

Data had been collected from various sources initially but data was imbalanced so we couldn't get good accuracy. There we collected water white noise data to balanced it up to some extent

### Data Arranger

The data that's been collected was in various directories but we needed it be in Alarm, Water and Other classes.

For that data_arranger script had been prepared which

- mixes up the data and provide us the refined data directory containing three classes
- also splits the data into 95% training and 5% testing
- lastly takes one sample testing directory of Alarm, Water and Other; as an inference

### Data Preprocessing

The data is in perfect form but we need to do a lot of work on it to feed it to the model, the data_preprocessor.py script has been used where;

- First we extract features by apply STFT to load in the data and MFCC to extract features
- Train set is then split into 85% and 15% validation
- After feature extraction the labels are converted into categorical.
- Features are standardized through sklearn standard scaler
- And augment via a custom augmented function.

### Final dataset can be found here

Here the its been divided into three directories, raw, balanced and features.

- **raw** contains the consecutive folders for the audio files of the data
- **balanced** contains the data in the form train, test and inference, in each of these sub directories we have Alarm, Water and Other
- **features** contains the features extracted using data preprocessing

https://drive.google.com/open?id=1yNeQqvHEEPm8eN5ullnM34XkU6KoV03M&usp=drive_fs

## Training

Since all the data had been preprocessed, training was smooth like butter. So train.py script;

- First loaded the data
- Data was balanced using SMOTE algorithm
- Model was trained on 10 epochs with batch size 32 using train set and validation set
- At least the model had been test on test set and results were computed.
- Model involves two 1D convolution layers followed by one 1D MaxPooling layers. The structure is similary to AlexNet but pin down a little bit and we're using 1D.
- After training on 30 epochs with batch size 32 we got 89% accuracy. Perfectly predicting water and alarm, suggesting a need to collect more data for others class.

Weights can be found at this url

https://drive.google.com/open?id=1V6ES-tPA48wDQueZg_MxMXmIZMX2CbBY&usp=drive_fs