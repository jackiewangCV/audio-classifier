import os
import shutil
from tqdm.auto import tqdm
import random

def count_files_in_directory(directory):
    total_files = 0
    class_files_count = {}

    for root, dirs, files in os.walk(directory):
        # Count all files in the current directory
        num_files = len(files)
        total_files += num_files

        # Get the class name (subdirectory name)
        if root != directory:  # Make sure not to count the root directory itself
            class_name = os.path.basename(root)
            class_files_count[class_name] = num_files

    return total_files, class_files_count

############################################################
########### Displaying Counts for source dir ##############
#############################################################
# Path to the data/raw directory
directory_path = 'data/raw'

print('\nDiscovered Raw files\n')

total_files, class_files_count = count_files_in_directory(directory_path)

print(f"Total number of files: {total_files}")
print("Number of files in each class:")
for class_name, num_files in class_files_count.items():
    print(f"{class_name}: {num_files}")

############################################################
########### Arranging files to train directory ##############
#############################################################

###### Define the categories and their corresponding sounds ##############
print("\nArranging files to data/train\n")

# Define the source and target directories
source_dir = 'data/raw'
train_dir = 'data/balanced/train'

# Define the target subdirectories
alarm_dir = os.path.join(train_dir, 'Alarm')
water_dir = os.path.join(train_dir, 'Water')
other_dir = os.path.join(train_dir, 'Other')

# Create the target directories if they don't exist
os.makedirs(alarm_dir, exist_ok=True)
os.makedirs(water_dir, exist_ok=True)
os.makedirs(other_dir, exist_ok=True)

# List of category names and their corresponding sounds
categories = {
    'Alarm': ['Alarm', 'Alarm_clock', 'Bell', 'Busy_signal', 'Car_alarm', 'Cellphone_buzz_vibrating_alert',
              'Fire_alarm', 'Siren', 'smoke_detector_smoke_alarm', 'Telephone_bell_ringing', 'audioset-smoke_alarm'],
    'Water': ['bathtub_filling_or_washing', 'Fill_with_liquid', 'Shower', 'Stream_river', 'toilet_flush',
              'Water', 'Water_tap_faucet', 'Waterfall', 'WaterWhiteNoise'],
    'Other': ['dishes_pots_and_pans', 'door', 'microwave_oven']
}

def copy_files_to_category():
    # Create a total count for the tqdm progress bar
    progress_bar = tqdm(total=total_files, desc="Copying files")

    # Walk through all files in the source directory
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # Identify the category of the file
            for category, sounds in categories.items():
                if any(sound in root for sound in sounds):
                    source_file = os.path.join(root, file)
                    target_dir = os.path.join(train_dir, category)
                    target_file = os.path.join(target_dir, file)
                    shutil.copy2(source_file, target_file)
                    progress_bar.update(1)
                    break  # Stop checking once the correct category is found

    progress_bar.close()

# Copy files to the respective directories

# copy_files_to_category()

############################################################
########### Displaying Counts for train dir ##############
#############################################################

total_files, class_files_count = count_files_in_directory(train_dir)

print(f"Total number of files: {total_files}")
print("Number of files in each class:")
for class_name, num_files in class_files_count.items():
    print(f"{class_name}: {num_files}")


############################################################
########### Sampling for test dir ##############
#############################################################

# Define the source and target directories
print('\nCollecting test sample\n')

train_dir = 'data/balanced/train'
test_dir = 'data/balanced/test'

# Define the subdirectories
categories = ['Alarm', 'Water', 'Other']

# Create the target directories if they don't exist
for category in categories:
    os.makedirs(os.path.join(test_dir, category), exist_ok=True)

def sample_files_to_test(train_dir, test_dir, sample_percentage):
    for category in categories:
        category_train_dir = os.path.join(train_dir, category)
        category_test_dir = os.path.join(test_dir, category)

        # Get list of all files in the category directory
        all_files = [f for f in os.listdir(category_train_dir) if os.path.isfile(os.path.join(category_train_dir, f))]

        # Determine the number of files to sample
        sample_size = max(1, int(len(all_files) * sample_percentage / 100))

        # Randomly sample files
        sampled_files = random.sample(all_files, sample_size)

        # Move sampled files to the test directory
        with tqdm(total=sample_size, desc=f"Sampling {category} files") as pbar:
            for file in sampled_files:
                source_file = os.path.join(category_train_dir, file)
                target_file = os.path.join(category_test_dir, file)
                shutil.move(source_file, target_file)
                pbar.update(1)


# Call the function to sample and move files

# sample_files_to_test(train_dir, test_dir, sample_percentage=5)

print("Files have been sampled and moved to the test directory successfully.")


############################################################
########### Sampling for Inference ##############
#############################################################
print('\nCollecting inference files\n')

# Define the source directories for each category
test_dir = 'data/balanced/test'
categories = ['Alarm', 'Water', 'Other']

# Define the target inference directory
inference_dir = 'data/balanced/inference'

# Create the inference directory if it doesn't exist
os.makedirs(inference_dir, exist_ok=True)

def move_files_for_inference(test_dir, inference_dir, categories):
    for category in categories:
        category_dir = os.path.join(test_dir, category)

        # Get list of all files in the category directory
        all_files = [f for f in os.listdir(category_dir) if os.path.isfile(os.path.join(category_dir, f))]

        if all_files:
            # Randomly select one file
            selected_file = random.choice(all_files)

            source_file = os.path.join(category_dir, selected_file)
            target_file = os.path.join(inference_dir, f"{category}.wav")

            # Move the selected file to the inference directory
            shutil.move(source_file, target_file)
            print(f"Moved {selected_file} from {category} to inference directory as {category}.wav")


# Call the function to move files for inference

# move_files_for_inference(test_dir, inference_dir, categories)
