from scipy.io import arff
import pandas as pd

# Citește fișierul ARFF
data, meta = arff.loadarff(r"D:\master\picsc\datasets\Scenario A2-ARFF\Scenario A2-ARFF\TimeBasedFeatures-Dataset-15s-NO-VPN.arff")

# Transformă într-un DataFrame
df = pd.DataFrame(data)

# Dacă datele sunt în bytes, decodifică-le (frecvent pentru string-uri)
for col in df.select_dtypes([object]):
    df[col] = df[col].str.decode('utf-8')

# Salvează în CSV
df.to_csv("features_timeout_15s_nonvpn.csv", index=False)
