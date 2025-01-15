import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter
from scipy.signal import argrelextrema
from sklearn.metrics import mean_absolute_error

pd.options.mode.chained_assignment = None


# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2


# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/01.data_processed.pkl")
df = df[df["label"] != "rest"]



# Calculate the resultant vector magnitude for accelerometer and gyroscope data
acc_r = (
    df["acc_x"] ** 2 
    + df["acc_y"] ** 2 
    + df["acc_z"] ** 2
)
gyr_r = (
    df["gyr_x"] ** 2 
    + df["gyr_y"] ** 2 
    + df["gyr_z"] ** 2
)

# Add resultant vectors to the dataframe
df["acc_r"] = np.sqrt(acc_r)
df["gyr_r"] = np.sqrt(gyr_r)

# --------------------------------------------------------------
# Split data
# --------------------------------------------------------------
bench_df = df[df["label"] == "bench"]
squat_df = df[df["label"] == "squat"]
row_df = df[df["label"] == "row"]
ohp_df = df[df["label"] == "ohp"]
dead_df = df[df["label"] == "dead"]

# --------------------------------------------------------------
# Visualize data to identify patterns
# --------------------------------------------------------------
plot_df = squat_df

# Plot accelerometer data for the first unique set

plot_df[plot_df ["set"] == plot_df ["set"].unique()[0]]["acc_x"].plot() 
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_y"].plot(label="Acc Y")
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_z"].plot(label="Acc Z")
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["acc_r"].plot(label="Acc R")

# Plot gyroscope data for the first unique set
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_x"].plot(label="Gyr X")
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_y"].plot(label="Gyr Y")
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_z"].plot(label="Gyr Z")
plot_df[plot_df["set"] == plot_df["set"].unique()[0]]["gyr_r"].plot(label="Gyr R")


# --------------------------------------------------------------
# Configure LowPassFilter
# --------------------------------------------------------------
fs = 1000 / 200
LowPass =  LowPassFilter()

# --------------------------------------------------------------
# Apply and tweak LowPassFilter
# --------------------------------------------------------------
# Define dataframes and sets
bench_set = bench_df[bench_df["set"] == bench_df["set"].unique()[0]]
squat_set = squat_df[squat_df["set"] == squat_df["set"].unique()[0]]
row_set = row_df[row_df["set"] == row_df["set"].unique()[0]]
ohp_set = ohp_df[ohp_df["set"] == ohp_df["set"].unique()[0]]
dead_set = dead_df[dead_df["set"] == dead_df["set"].unique()[0]]


column = "acc_r"
LowPass.low_pass_filter(bench_set, col=column, sampling_frequency=fs, cutoff_frequency=0.4, order=10)[column + "_lowpass"].plot()

# --------------------------------------------------------------
# Create function to count repetitions
# --------------------------------------------------------------

def count_reps(dataset, cutoff=0.4,order=10,column="acc_r"):
    data = LowPass.low_pass_filter(dataset, col=column, sampling_frequency=fs, cutoff_frequency=cutoff, order=order)
    indexes = argrelextrema(data[column + "_lowpass"].values, np.greater)
    peaks = data.iloc[indexes]
    
    # Plot the filtered data and peaks
    fig, ax = plt.subplots()
    plt.plot(data[f"{column}_lowpass"], label=f"{column} Lowpass")
    plt.plot(peaks[f"{column}_lowpass"], "o", color="red", label="Peaks")
    ax.set_ylabel(f"{column} Lowpass")
    
    # Extract exercise and category labels
    exercise = dataset["label"].iloc[0].title()
    category = dataset["category"].iloc[0].title()
    
    # Add title and legend
    plt.title(f"{category} {exercise}: {len(peaks)} Reps")
    plt.legend()
    plt.show()
    
    
    
    return len(peaks)

# Call the function for different datasets
count_reps(bench_set, cutoff=0.4)
count_reps(squat_set, cutoff=0.35)
count_reps(row_set, cutoff=0.65, column="gyr_x")
count_reps(ohp_set, cutoff=0.35)
count_reps(dead_set, cutoff=0.4)


# --------------------------------------------------------------
# Create benchmark dataframe
# --------------------------------------------------------------
# Add a 'reps' column to the dataframe based on the 'category'
df["reps"] = df["category"].apply(lambda x: 5 if x == "heavy" else 10)

# Group by label, category, and set to compute the maximum reps
rep_df = df.groupby(["label", "category", "set"])["reps"].max().reset_index()

# Add a column to store predicted reps
rep_df["reps_pred"] = 0

# Loop through each unique set and predict the reps
for s in df["set"].unique():
    # Filter the dataframe for the current set
    subset = df[df["set"] == s]
    column = "acc_r"
    cutoff = 0.4  # Default cutoff frequency
    
    # Adjust cutoff and column based on the label
    if subset["label"].iloc[0] == "squat":
        cutoff = 0.35
    elif subset["label"].iloc[0] == "row":
        cutoff = 0.65
        column = "gyr_x"
    elif subset["label"].iloc[0] == "ohp":
        cutoff = 0.35

    # Count the reps using the count_reps function
    reps = count_reps(subset, cutoff=cutoff, column=column)
    
    # Update the predicted reps in the grouped dataframe
    rep_df.loc[rep_df["set"] == s, "reps_pred"] = reps

# Display the resulting dataframe
rep_df


# --------------------------------------------------------------
# Evaluate the results
# --------------------------------------------------------------
error = mean_absolute_error(rep_df["reps"], rep_df["reps_pred"]).round(2)

rep_df.groupby(["label", "category"])[["reps", "reps_pred"]].mean().plot.bar()
