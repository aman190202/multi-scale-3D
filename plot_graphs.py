# Confidence: 100%
import pandas as pd
import matplotlib.pyplot as plt

# === Load CSV ===
df = pd.read_csv("pairwise_pose_errors.csv")

# === 1. Rotation error vs GT rotation angle ===
plt.figure()
plt.scatter(df["gt_rotation_deg"], df["rotation_error_deg"], s=15, alpha=0.7)
plt.xlabel("Ground Truth Rotation (deg)")
plt.ylabel("Rotation Error (deg)")
plt.title("Rotation Error vs. GT Rotation Angle")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotation_error_vs_gt_rotation.png", dpi=300)

# === 2. Translation direction error vs GT rotation angle ===
plt.figure()
plt.scatter(df["gt_rotation_deg"], df["translation_dir_error_deg"], s=15, alpha=0.7, color="orange")
plt.xlabel("Ground Truth Rotation (deg)")
plt.ylabel("Translation Direction Error (deg)")
plt.title("Translation Direction Error vs. GT Rotation Angle")
plt.grid(True)
plt.tight_layout()
plt.savefig("translation_error_vs_gt_rotation.png", dpi=300)

# === 3. Rotation error vs GT translation distance ===
plt.figure()
plt.scatter(df["gt_translation_distance"], df["rotation_error_deg"], s=15, alpha=0.7, color="green")
plt.xlabel("Ground Truth Translation Distance")
plt.ylabel("Rotation Error (deg)")
plt.title("Rotation Error vs. GT Translation Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("rotation_error_vs_gt_translation.png", dpi=300)

# === 4. Translation direction error vs GT translation distance ===
plt.figure()
plt.scatter(df["gt_translation_distance"], df["translation_dir_error_deg"], s=15, alpha=0.7, color="red")
plt.xlabel("Ground Truth Translation Distance")
plt.ylabel("Translation Direction Error (deg)")
plt.title("Translation Direction Error vs. GT Translation Distance")
plt.grid(True)
plt.tight_layout()
plt.savefig("translation_error_vs_gt_translation.png", dpi=300)

print("Plots saved: 4 PNG files generated.")