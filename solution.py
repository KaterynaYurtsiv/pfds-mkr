import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DB_USER = "student"
DB_PASSWORD = "student"
DB_HOST = "localhost"
DB_PORT = 3306
DB_NAME = "meteo"

connection_string = (
    f"mysql+mysqlconnector://{DB_USER}:{DB_PASSWORD}"
    f"@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

engine = create_engine(connection_string)

raw_df = pd.read_sql("SELECT * FROM observations", engine)

print("=== BLOCK 1. NUMPY ===")

temperature = raw_df["temperature_c"].to_numpy(dtype=float)
humidity = raw_df["humidity_pct"].to_numpy(dtype=float)
wind = raw_df["wind_speed_ms"].to_numpy(dtype=float)

apparent_temperature = temperature - (100 - humidity) / 5

temperature_np = np.where(
    (temperature > 60) | (temperature < -60),
    np.nan,
    temperature
)

wind_np = np.where(
    wind > 100,
    np.nan,
    wind
)

valid_temp = temperature_np[~np.isnan(temperature_np)]

temp_mean = np.nansum(temperature_np) / np.sum(~np.isnan(temperature_np))
temp_median = np.nanmedian(temperature_np)
temp_std = np.sqrt(
    np.nansum((valid_temp - temp_mean) ** 2) / valid_temp.size
)

frost_count = np.sum(temperature_np < 0)
hot_count = np.sum(temperature_np > 30)

max_idx = np.nanargmax(temperature_np)
min_idx = np.nanargmin(temperature_np)

print(f"Mean temperature: {temp_mean:.2f}")
print(f"Median temperature: {temp_median:.2f}")
print(f"Standard deviation: {temp_std:.2f}")

print(f"Frost observations (T < 0): {frost_count}")
print(f"Hot observations (T > 30): {hot_count}")

print("Maximum temperature:")
print(f"obs_id: {raw_df.loc[max_idx, 'obs_id']}")
print(f"datetime: {raw_df.loc[max_idx, 'datetime']}")
print(f"temperature: {temperature_np[max_idx]:.2f}")

print("Minimum temperature:")
print(f"obs_id: {raw_df.loc[min_idx, 'obs_id']}")
print(f"datetime: {raw_df.loc[min_idx, 'datetime']}")
print(f"temperature: {temperature_np[min_idx]:.2f}")

print("\n=== BLOCK 2. PANDAS CLEANING ===")

df = raw_df.copy()

rows_before = len(df)
missing_before = df.isna().sum().sum()

duplicates_count = df.duplicated().sum()
df = df.drop_duplicates()

df["datetime"] = pd.to_datetime(df["datetime"])
df = df.set_index("datetime")

df["month"] = df.index.month

humidity_missing_before = df["humidity_pct"].isna().sum()

df["humidity_pct"] = df.groupby(["city", "month"])["humidity_pct"] \
    .transform(lambda s: s.fillna(s.median()))

humidity_missing_after = df["humidity_pct"].isna().sum()
humidity_filled = humidity_missing_before - humidity_missing_after

rows_before_outliers = len(df)

df = df[
    (df["temperature_c"].between(-60, 60)) &
    (
        df["wind_speed_ms"].isna() |
        df["wind_speed_ms"].between(0, 60)
    )
]

outliers_removed = rows_before_outliers - len(df)
rows_after = len(df)

print(f"Rows before cleaning: {rows_before}")
print(f"Rows after cleaning: {rows_after}")
print(f"Total missing values before cleaning: {missing_before}")
print(f"Duplicates removed: {duplicates_count}")
print(f"Humidity NaN filled: {humidity_filled}")
print(f"Outliers removed: {outliers_removed}")

print("\nCleaned data info:")
print(df.info())

print("\nCleaned data description:")
print(df.describe())

print("\n=== BLOCK 3. PANDAS ANALYTICS ===")

mean_temp_by_city = df.groupby("city")["temperature_c"].mean()

warmest_city = mean_temp_by_city.idxmax()
coldest_city = mean_temp_by_city.idxmin()

print("\nAverage temperature by city:")
print(mean_temp_by_city)

print(f"\nWarmest city: {warmest_city}")
print(f"Coldest city: {coldest_city}")

precipitation_by_city = df.groupby("city")["precipitation_mm"].sum()

wettest_city = precipitation_by_city.idxmax()

print("\nTotal precipitation by city:")
print(precipitation_by_city)

print(f"\nWettest city: {wettest_city}")

monthly_temperature = df["temperature_c"].resample("ME").mean()

print("\nMonthly average temperature:")
print(monthly_temperature)

pivot_table = df.pivot_table(
    values="temperature_c",
    index="city",
    columns="month",
    aggfunc="mean"
)

print("\nPivot table (city x month):")
print(pivot_table)

rainy_days = df[df["precipitation_mm"] > 5].copy()

rainy_days["date"] = rainy_days.index.date

rainy_days_count = rainy_days.groupby("city")["date"].nunique()

print("\nDays with precipitation > 5 mm:")
print(rainy_days_count)

monthly_data = df["temperature_c"].resample("ME").mean().to_frame(name="temp")

monthly_data["year"] = monthly_data.index.year
monthly_data["month"] = monthly_data.index.month

monthly_norm = monthly_data.groupby("month")["temp"].mean()

monthly_data["norm"] = monthly_data["month"].map(monthly_norm)

monthly_data["deviation"] = (
    monthly_data["temp"] - monthly_data["norm"]
)

monthly_data["abs_deviation"] = (
    monthly_data["deviation"].abs()
)

anomaly_idx = monthly_data["abs_deviation"].idxmax()

anomaly_row = monthly_data.loc[anomaly_idx]

print("\nAnomalous month:")
print(f"Year: {anomaly_row['year']}")
print(f"Month: {anomaly_row['month']}")
print(f"Temperature deviation: {anomaly_row['deviation']:.2f} °C")

if anomaly_row["deviation"] > 0:
    print("Type: Heat wave")
else:
    print("Type: Cold wave")

print("\n=== BLOCK 4. MATPLOTLIB PLOTS ===")

os.makedirs("plots", exist_ok=True)

# 1. Line plot — monthly temperature for 3 cities
selected_cities = ["Київ", "Львів", "Одеса"]

plt.figure(figsize=(10, 6))

for city in selected_cities:
    city_monthly_temp = df[df["city"] == city]["temperature_c"].resample("ME").mean()
    plt.plot(city_monthly_temp.index, city_monthly_temp.values, marker="o", label=city)

plt.title("Monthly average temperature in selected cities")
plt.xlabel("Month")
plt.ylabel("Temperature, °C")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("plots/01_monthly_temperature.png", dpi=300)
plt.close()


# 2. Bar plot — total precipitation by city
plt.figure(figsize=(8, 6))

precipitation_by_city.plot(kind="bar")

plt.title("Total precipitation by city")
plt.xlabel("City")
plt.ylabel("Precipitation, mm")
plt.tight_layout()
plt.savefig("plots/02_precipitation_by_city.png", dpi=300)
plt.close()


# 3. Histogram — temperature distribution with mean and median
plt.figure(figsize=(8, 6))

mean_temperature = df["temperature_c"].mean()
median_temperature = df["temperature_c"].median()

plt.hist(df["temperature_c"], bins=30, edgecolor="black")
plt.axvline(mean_temperature, linestyle="--", label=f"Mean: {mean_temperature:.2f}")
plt.axvline(median_temperature, linestyle="-", label=f"Median: {median_temperature:.2f}")

plt.title("Temperature distribution")
plt.xlabel("Temperature, °C")
plt.ylabel("Number of observations")
plt.legend()
plt.tight_layout()
plt.savefig("plots/03_temperature_histogram.png", dpi=300)
plt.close()


# 4. Heatmap — city x month average temperature
plt.figure(figsize=(10, 6))

plt.imshow(pivot_table, aspect="auto")

plt.title("Average temperature heatmap: city x month")
plt.xlabel("Month")
plt.ylabel("City")
plt.colorbar(label="Temperature, °C")

plt.xticks(
    ticks=range(len(pivot_table.columns)),
    labels=pivot_table.columns
)

plt.yticks(
    ticks=range(len(pivot_table.index)),
    labels=pivot_table.index
)

plt.tight_layout()
plt.savefig("plots/04_city_month_heatmap.png", dpi=300)
plt.close()

print("Plots saved to the 'plots' folder.")

"""
Висновки:

Найтеплішим містом у наборі даних виявився Київ, а найхолоднішим – Дніпро.
Також Київ має найбільшу сумарну кількість опадів, що свідчить про більш
вологий клімат у порівнянні з іншими містами.

Динаміка температур чітко демонструє сезонність клімату. Найвищі
температури спостерігаються наприкінці весни та влітку, а найнижчі –
у зимовий період.

Аномальним місяцем було визначено травень 2023 року, у якому зафіксовано
відхилення температури приблизно на -4.25 °C від кліматичної норми.
Цю аномалію можна класифікувати як хвилю холоду.

Теплова карта показала, що Київ у більшості місяців має вищі середні
температури, тоді як Дніпро та Харків характеризуються холоднішими зимами.

Під час роботи набір даних потребував попереднього очищення, оскільки
містив дублікати, пропущені значення та фізичні викиди. Проведене очищення
дозволило отримати більш точні та надійні результати аналізу.
"""