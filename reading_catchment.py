import os


# Define the folder path
folder_path = '/Users/justusnogel/Documents/applied_landsurface_modelling/catchment_donau/'

# Define the output file path
output_file_path = '/Users/justusnogel/Documents/applied_landsurface_modelling/catchment_coordinates_donau.txt'

# Open the output file in writing mode
with open(output_file_path, "w") as outfile:
    outfile.write("GRDC,latitude,longitude,catchment_area\n")  # Write header to the output file

    # Loop through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # Check if the file is a text file
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="windows-1252") as file:
                lines = file.readlines()
                latitude_line = lines[12].strip().split(":")[1].strip()
                longitude_line = lines[13].strip().split(":")[1].strip()
                area_line = lines[14].strip().split(":")[1].strip()

            # Extract GRDC number from filename
            grdc_number = filename.split("_")[0]

            # Check if catchment area is above 300
            if float(area_line) < 300 and 4.75 <= float(longitude_line) <= 15.25 and 44.75 <= float(latitude_line) <= 55.25:
                print(longitude_line)
                # Write latitude, longitude, and GRDC number to the output file
                outfile.write("{},{},{},{}\n".format(grdc_number, latitude_line, longitude_line, area_line))

# Print message indicating completion
print("Latitude and longitude saved to", output_file_path)
