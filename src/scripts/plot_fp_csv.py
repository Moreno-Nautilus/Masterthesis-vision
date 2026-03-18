import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

df = pd.read_csv(sys.argv[1])

out_dir = Path("plots")
out_dir.mkdir(exist_ok=True)

# --- Camera frame ---
plt.figure()
plt.plot(df["raw_z"], label="raw_z")
plt.plot(df["inv_z"], label="inv_z")
plt.legend()
plt.title("Camera frame Z")
plt.grid()
plt.savefig(out_dir / "camera_z.png")   # ✅ SAVE
plt.close()

# --- Base frame ---
plt.figure()
plt.plot(df["base_raw_z"], label="base_raw_z")
plt.plot(df["base_inv_z"], label="base_inv_z")
plt.legend()
plt.title("Base frame Z")
plt.grid()
plt.savefig(out_dir / "base_z.png")     # ✅ SAVE
plt.close()

print(f"Saved plots to: {out_dir.resolve()}")