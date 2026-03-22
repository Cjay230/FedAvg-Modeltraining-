import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

ROOT = Path(r"C:\Users\user\Desktop\VIP\kaust data\dataset_by_city_labeled\clients\fl_ready")

files = [
    "jeddah_train.csv",
    "jeddah_test.csv",
    "kaust_train.csv",
    "kaust_test.csv",
    "kz_train.csv",
    "kz_test.csv",
    "mekkah_train.csv",
    "mekkah_test.csv"
]

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1, lon1, lat2, lon2 = map(radians,[lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1-a))
    return R * c

for f in files:

    print("Processing", f)

    df = pd.read_csv(ROOT / f)

    df = df.sort_values("Time_seconds").reset_index(drop=True)

    distances = [0]
    time_deltas = [0]
    speeds = [0]
    accels = [0]

    for i in range(1,len(df)):

        lat1 = df.loc[i-1,"Latitude"]
        lon1 = df.loc[i-1,"Longitude"]

        lat2 = df.loc[i,"Latitude"]
        lon2 = df.loc[i,"Longitude"]

        t1 = df.loc[i-1,"Time_seconds"]
        t2 = df.loc[i,"Time_seconds"]

        d = haversine(lat1,lon1,lat2,lon2)
        dt = t2 - t1

        if dt == 0:
            v = 0
        else:
            v = d/dt

        distances.append(d)
        time_deltas.append(dt)
        speeds.append(v)

        if i == 1:
            accels.append(0)
        else:
            dv = v - speeds[i-1]
            accels.append(dv / dt if dt != 0 else 0)

    df["distance"] = distances
    df["time_delta"] = time_deltas
    df["speed"] = speeds
    df["acceleration"] = accels

    df.to_csv(ROOT / f, index=False)

print("Feature engineering complete.")