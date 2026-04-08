import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results.csv")

avg_cost = df["cost"].mean()
avg_peak = df["peak"].mean()

print("Average Cost:", avg_cost)
print("Average Peak:", avg_peak)

plt.figure()
plt.bar(df["model"], df["cost"])
plt.title("Cost per Model")
plt.xlabel("Model")
plt.ylabel("Cost")
plt.savefig("cost_per_model.png")

plt.figure()
plt.bar(df["model"], df["peak"])
plt.title("Peak Violations per Model")
plt.xlabel("Model")
plt.ylabel("Peak")
plt.savefig("peak_per_model.png")

plt.figure()
plt.scatter(df["peak"], df["cost"])

for i in range(len(df)):
    plt.text(df["peak"][i], df["cost"][i], f"M{i}")

plt.xlabel("Peak Violations")
plt.ylabel("Cost")
plt.title("Cost vs Peak Trade-off (Models)")
plt.savefig("cost_vs_peak.png")

plt.figure()
plt.scatter(df["peak"], df["cost"], label="Models")
plt.axhline(avg_cost, linestyle="--", label="Avg Cost")
plt.axvline(avg_peak, linestyle="--", label="Avg Peak")

plt.xlabel("Peak")
plt.ylabel("Cost")
plt.title("Average Performance")
plt.legend()
plt.savefig("average_performance.png")

print("Graphs saved successfully!")