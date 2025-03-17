import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 5000

# Generate synthetic features
data = {
    "Packet_Size": np.random.randint(50, 1500, n_samples),  # Packet size in bytes
    "Connection_Duration": np.random.uniform(1, 60, n_samples),  # Connection duration in seconds
    "Failed_Logins": np.random.randint(0, 10, n_samples),  # Number of failed login attempts
    "CPU_Usage": np.random.uniform(0, 100, n_samples),  # CPU usage percentage
    "Memory_Usage": np.random.uniform(0, 100, n_samples),  # Memory usage percentage
    "File_Size": np.random.randint(1, 1000, n_samples),  # File size in KB
    "Outgoing_Traffic": np.random.uniform(0, 1000, n_samples),  # Outgoing traffic in MB
    "Incoming_Traffic": np.random.uniform(0, 1000, n_samples),  # Incoming traffic in MB
    "URL_Length": np.random.randint(10, 200, n_samples),  # Length of URLs in characters
    "Email_Content_Length": np.random.randint(50, 500, n_samples),  # Length of email content in words
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Define the target variable (label)
# Malicious instances are generated based on certain conditions
df['label'] = 0  # Initialize all as benign
malicious_conditions = (
    (df["Failed_Logins"] > 5) |  # High number of failed logins (Malware Detection)
    (df["CPU_Usage"] > 80) |  # High CPU usage (Threat Analysis)
    (df["Outgoing_Traffic"] > 800) |  # High outgoing traffic (Intrusion Detection)
    (df["URL_Length"] > 150) |  # Long URLs (Phishing Detection)
    (df["Email_Content_Length"] > 400)  # Long email content (Phishing Detection)
)
df.loc[malicious_conditions, 'label'] = 1  # Mark as malicious

# Shuffle the dataset
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save the dataset to a CSV file
df.to_csv("cybersecurity_dataset.csv", index=False)

# Display the first few rows of the dataset
print(df.head())