import pandas as pd

df = pd.read_csv("data/2020-263907-one_axis.csv", skiprows=2)

df = df[df["GHI"] != -9999]

ghi = df["GHI"].values[:168]

solar = ghi / 20

solar_df = pd.DataFrame({"solar": solar})
solar_df.to_csv("data/solar.csv", index=False)

print("7-day solar.csv created")