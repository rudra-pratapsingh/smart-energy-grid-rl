import pandas as pd

print("Loading dataset...")

df = pd.read_csv(
    "data/household_power_consumption.txt",
    sep=";",
    low_memory=False,
    na_values=["?"]
)

df = df.dropna()

df["datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"])

df["Global_active_power"] = df["Global_active_power"].astype(float)

df.set_index("datetime", inplace=True)

hourly = df["Global_active_power"].resample("h").mean()

hourly = hourly[100:148]

load_df = pd.DataFrame({"load": hourly.values})
load_df.to_csv("data/load.csv", index=False)

print("load.csv created successfully!")
