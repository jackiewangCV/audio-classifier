# Audio Classification

## Data 

### Data Arranger

The data that's been collected was in various directories but we needed it be in Alarm, Water and Other classes.

For that data_arranger script had been prepared which

- mixes up the data and provide us the refined data directory containing three classes
- also splits the data into 95% training and 5% testing
- lastly takes one sample testing directory of Alarm, Water and Other; as an inference

The resultant dataset can found at this URL:

https://drive.google.com/open?id=1yNeQqvHEEPm8eN5ullnM34XkU6KoV03M&usp=drive_fs

### Data Preprocessing

The data is in perfect form but we need to do a lot of work on it to feed it to the model, the make_data_16k.py script has been used where;

- First we extract features by apply STFT to load in the data and MFCC to extract features
- Train set is then split into 85% and 15% validation
- After feature extraction the labels are converted into categorical.
- Features are standardized through sklearn standard scaler
- And augment via a custom augmented function.

The resultant preprocessed data can be found at this url:

https://drive.google.com/drive/folders/1vrY63UxbZ4AEnwVsatmeTlOr5VqoOknZ?usp=sharing

## Training

Since all the data had been preprocessed, training was smooth like butter. So train_16k.py script;

- First loaded the data
- Data was balanced using SMOTE algorithm
- Model was trained on 10 epochs with batch size 32 using train set and validation set
- At least the model had been test on test set and results were computed.
- Model had been changed to Residual Connection as they've proven good results in proven. It seems more reliable afterwards, what it means is that you can trust that everytime you hit the run button you will get 80% 81% accuracy
- Many experiment were conducted the highest accuracy so far was 80% 81%.
- Experiment 8 and Experiment 12 turns out to be reliable choice

