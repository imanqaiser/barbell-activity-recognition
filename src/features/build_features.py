import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add the `src` directory to the Python path
sys.path.append(os.path.abspath("../../src/features"))

# Now import your module
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------
df = pd.read_pickle("../../data/interim/02.outliers_removed_chauvenets.pkl")

predictor_columns = list(df.columns[:6])
# Plot settings
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# --------------------------------------------------------------
# Dealing with missing values (imputation)
# --------------------------------------------------------------
for col in predictor_columns:
    df[col] = df[col].interpolate()

df.info()

# --------------------------------------------------------------
# Calculating set duration
# --------------------------------------------------------------


# Calculate and assign durations for all sets
for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]
    duration = stop - start
    df.loc[df["set"] == s, "duration"] = duration.seconds

# Calculate the mean duration for each category
duration_df = df.groupby(["category"])["duration"].mean()

# Normalize durations for specific categories
duration_df.iloc[0] / 5 # 5 reps in heavy sets
duration_df.iloc[1] / 10 # 10 repes in medium sets

((duration_df.iloc[0] / 5 ) + (duration_df.iloc[1] / 10) )/2 #average duration


# --------------------------------------------------------------
# Butterworth lowpass filter
# --------------------------------------------------------------

df_lowpass = df.copy()
LowPass = LowPassFilter()

# Sampling frequency and cutoff frequency
fs = 1000 / 200
cutoff = 1.3

# Apply low-pass filter to "acc_y"
df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)

# Select subset for a specific set
subset = df_lowpass[df_lowpass["set"] == 10]
print(subset["label"].iloc[0])

# Plot raw and filtered data
fig, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="butterworth filter")

# Add legends
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)

# Apply low-pass filter to all predictor columns
for col in predictor_columns:
    df_lowpass = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpass[col + "_lowpass"]
    del df_lowpass[col + "_lowpass"]


# --------------------------------------------------------------
# Principal component analysis PCA
# --------------------------------------------------------------

#PCA is a technique used in machine learning to reduce the complexity of data by transforming the data into a new set of variables called principal components. This transformation is done in such a way that the new set of variables captures the most amount of information from the original data set, while reducing the number of variables necessary.

# Create a copy of the low-pass filtered dataframe
df_pca = df_lowpass.copy()

# Initialize Principal Component Analysis
PCA = PrincipalComponentAnalysis()

# Determine explained variance for principal components
pc_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)

# Plot the explained variance
plt.figure(figsize=(10, 10))
plt.plot(range(1, len(predictor_columns) + 1), pc_values)
plt.xlabel("Principal Component Number")
plt.ylabel("Explained Variance")
plt.title("Explained Variance by Principal Components")
plt.show()

# Apply PCA to reduce dimensions to 3 components -> chosen as it is the "elbow" number
df_pca = PCA.apply_pca(df_pca, predictor_columns, 3)

# Select a subset for a specific set
subset = df_pca[df_pca["set"] == 35]

# Plot the first three principal components
subset[["pca_1", "pca_2", "pca_3"]].plot()



# --------------------------------------------------------------
# Sum of squares attributes
# --------------------------------------------------------------
# Create a copy of the PCA dataframe for squared calculations
df_squared = df_pca.copy()

# Calculate the resultant vector magnitude for accelerometer and gyroscope data
acc_r = (
    df_squared["acc_x"] ** 2 
    + df_squared["acc_y"] ** 2 
    + df_squared["acc_z"] ** 2
)
gyr_r = (
    df_squared["gyr_x"] ** 2 
    + df_squared["gyr_y"] ** 2 
    + df_squared["gyr_z"] ** 2
)

# Add resultant vectors to the dataframe
df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)

# Select a subset for a specific set
subset = df_squared[df_squared["set"] == 14]

# Plot the resultant vectors as subplots
subset[["acc_r", "gyr_r"]].plot(
    subplots=True, 
    figsize=(10, 6), 
    title="Resultant Vectors for Accelerometer and Gyroscope (Set 14)"
)
plt.xlabel("Samples")
plt.show()

# Display the updated dataframe
df_squared


# --------------------------------------------------------------
# Temporal abstraction
# --------------------------------------------------------------
#Rolling Windows for Statistical Feature 
df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()
predictor_columns = predictor_columns + ["acc_r", "gyr_r"]
ws = int(1000 / 200)

df_temporal_list = []
for s in df_temporal["set"].unique():
    subset = df_temporal [df_temporal ["set"] == s].copy()
    for col in predictor_columns:
        subset = NumAbs. abstract_numerical (subset, [col], ws, "mean") 
        subset = NumAbs.abstract_numerical (subset, [col], ws, "std")
    df_temporal_list.append(subset)
    
df_temporal = pd.concat(df_temporal_list)

subset [["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot() 
subset [["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

df_temporal.info()


# --------------------------------------------------------------
# Frequency features
# --------------------------------------------------------------

df_freq = df_temporal.copy().reset_index() 
FreqAbs = FourierTransformation()

fs= int(1000 / 200)
ws = int(2800 / 200) #window size deals with the avg length of a set, found ^^^

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# Visualize results for a specific set
subset = df_freq[df_freq["set"] == 15]

# Plot the raw accelerometer Y-axis data
subset[["acc_y"]].plot()

# Plot frequency-related columns for analysis
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot(

)


# Apply Fourier transformations to all unique sets
df_freq_list = []
for s in df_freq["set"].unique():
    print(f"Applying Fourier transformations to set {s}")

    # Reset and copy the subset for the current set
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()

    # Apply frequency abstraction
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)

    # Append the processed subset to the list
    df_freq_list.append(subset)

# Concatenate the results and set index to epoch (ms)
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)


# --------------------------------------------------------------
# Dealing with overlapping windows
# --------------------------------------------------------------
#Because of rolling windows, higly correleated values. Could cause overfitting !
df_freq = df_freq.dropna()
#Since we have lots of data, can afford a 50% only overlap
df_freq = df_freq.iloc[::2] # Gets every other row

# --------------------------------------------------------------
# Clustering
# --------------------------------------------------------------
df_cluster = df_freq.copy()
cluster_columns = ["acc_x", "acc_y", "acc_z"]

# Determine the optimal number of clusters
k_values = range(2, 10)
inertias = []

for k in k_values:
    subset = df_cluster[cluster_columns]
    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)
    inertias.append(kmeans.inertia_) #store these to figure out ideal value of K via elbow method

# Plot the sum of squared distances for different k values -> Elbow Method
plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("k")
plt.ylabel("Sum of squared distances")
plt.show()

# Perform clustering with 5 clusters
kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# Plot clusters
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# Plot accelerometer data to compare
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")

for label in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == label]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=label)

ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()



# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
df_cluster.to_pickle("../../data/interim/03.data_features.pkl")