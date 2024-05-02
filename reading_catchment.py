import os
import shutil

# Define the main folder path containing folders for each river
main_folder_path = '/Users/simon/Documents/Master/Semester_4/Applied_Landsurface_Modelling/data/'

# Define the destination folder path
destination_folder_path = '/Users/simon/Documents/Master/Semester_4/Applied_Landsurface_Modelling/selection/'

# Initialize counter for extracted files
total_extracted_files_count = 0

# Loop through each main folder
for river_folder in os.listdir(main_folder_path):
    river_folder_path = os.path.join(main_folder_path, river_folder)

    # Check if the current item is a directory
    if os.path.isdir(river_folder_path):
        # Initialize counter for extracted files for the current river
        extracted_files_count = 0

        # Loop through all files in the current river folder
        for filename in os.listdir(river_folder_path):
            if filename.endswith(".txt"):  # Check if the file is a text file
                file_path = os.path.join(river_folder_path, filename)

                with open(file_path, "r", encoding="windows-1252") as file:
                    lines = file.readlines()
                    # Extract latitude, longitude, and catchment area from specific lines
                    latitude_line = [line.strip().split(":")[1].strip() for line in lines if "Latitude" in line][0]
                    longitude_line = [line.strip().split(":")[1].strip() for line in lines if "Longitude" in line][0]
                    area_line = [line.strip().split(":")[1].strip() for line in lines if "Catchment area" in line][0]

                    # Check if catchment area is above 300 and within specified latitude and longitude range
                    if (float(area_line) < 300 and
                            4.75 <= float(longitude_line) <= 15.25 and
                            44.75 <= float(latitude_line) <= 55.25):
                        # Copy the file to the destination folder
                        shutil.copy(file_path, destination_folder_path)
                        extracted_files_count += 1
                        total_extracted_files_count += 1

        # Print message indicating completion and number of extracted files for the current river
        print(f"For river {river_folder}, {extracted_files_count} files copied to {destination_folder_path}")

# Print message indicating completion and total number of extracted files
print("Total files copied:", total_extracted_files_count)
print("Files copied to", destination_folder_path)
