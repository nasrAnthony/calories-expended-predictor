import csv

target_file = ".\\60secUpDownMotion.csv" 
output_file = ".\\output_no_motion_data.csv"

# Open the input CSV file for reading
with open(target_file, "r") as f:
    lines = f.readlines()

# Open a new CSV file for writing
with open(output_file, "w", newline="") as outfile:
    writer = csv.writer(outfile)
    # Write the header
    writer.writerow(['acc_x', 'acc_y', 'acc_z'])
    count = 0
    row_data = []
    for line in lines:
        if line.strip() == "": #skip "\n"
            continue
        else:
            data = line.strip().split(",")[1]
            match count:
                case 0:
                    row_data.append(data)
                    count += 1
                case 1:
                    #remove gravity bias. 
                    data_float = float(data)
                    data_float = data_float - 0.808
                    row_data.append(data_float)
                    count += 1
                case 2:
                    row_data.append(data)
                    writer.writerow(row_data)
                    row_data = []
                    count = 0


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
file_path = output_file

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Plot acc_x, acc_y, and acc_z
plt.figure(figsize=(10, 6))

# Plot acc_x
plt.plot(df['acc_x'], label='acc_x')

# Plot acc_y
plt.plot(df['acc_y'], label='acc_y')

# Plot acc_z
plt.plot(df['acc_z'], label='acc_z')

# Add titles and labels
plt.title('Acceleration Data (X, Y, Z)')
plt.xlabel('Sample')
plt.ylabel('Acceleration')
plt.legend()

# Show the plot
plt.show()

