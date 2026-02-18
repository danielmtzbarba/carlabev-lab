
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import PchipInterpolator

# ===========================
# 🔧 Configuration
# ===========================
ROOT = "/home/danielmtz/Data/results/carlabev/runs_mask_continuous"

# Global Plot Style: "Publication Quality"
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size": 14,
    "axes.titlesize": 20,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "legend.title_fontsize": 16,
    "figure.figsize": (12, 8),
    "lines.linewidth": 2.5,
    "lines.markersize": 8,
    "grid.alpha": 0.5,
})

# ===========================
# 🛠 Helpers
# ===========================

def parse_experiment_name(exp_name: str) -> dict:
    """
    Parse experiment folder name.
    Example: exp-1_cnn-ppo_act-discrete_traffic-off_input-rgb_rwd-shaping_curr-off_fovmask-off
    """
    meta = {
        "experiment": exp_name,
        "exp_id": None,
        "algo": None,
        "action_space": None,
        "traffic": None,
        "input": None,
        "reward": None,
        "curriculum": None,
        "fov_mask": None,
    }

    parts = exp_name.split("_")
    for p in parts:
        if p.startswith("exp-"):
            try:
                meta["exp_id"] = int(p.split("-")[1])
            except ValueError:
                meta["exp_id"] = None
        elif "cnn-ppo" in p:
            meta["algo"] = "cnn-ppo"
        elif p.startswith("act-"):
            meta["action_space"] = p.split("act-")[1]
        elif p.startswith("traffic-"):
            meta["traffic"] = p.split("traffic-")[1]
        elif p.startswith("input-"):
            meta["input"] = p.split("input-")[1]
        elif p.startswith("rwd-"):
            meta["reward"] = p.split("rwd-")[1]
        elif p.startswith("curr-"):
            meta["curriculum"] = p.split("curr-")[1]
        elif p.startswith("fovmask-"):
            # Special handling for user request: "mask_on" / "mask_off" style
            p_val = p.split("fovmask-")[1]
            meta["fov_mask"] = f"mask_{p_val}"

    return meta

def load_data(root: str):
    """
    Walks through ROOT and loads:
      1. ppo-eval-final-1000.npy -> df_eval
      2. benchmark_results.csv   -> df_benchmark
    """
    all_eval_rows = []
    all_bench_rows = []

    if not os.path.exists(root):
        print(f"❌ Root directory not found: {root}")
        return pd.DataFrame(), pd.DataFrame()

    print(f"📂 Scanning {root}...")

    for exp in sorted(os.listdir(root)):
        exp_dir = os.path.join(root, exp)
        if not os.path.isdir(exp_dir):
            continue

        meta = parse_experiment_name(exp)
        if meta["exp_id"] is None:
             continue

        # 1. Load Eval Results (.npy)
        eval_file = os.path.join(exp_dir, "ppo-eval-final-1000.npy")
        if os.path.exists(eval_file):
            try:
                data = np.load(eval_file, allow_pickle=True).item()
                row = {**meta, **data}
                all_eval_rows.append(row)
            except Exception as e:
                print(f"  ⚠️ Error loading {eval_file}: {e}")

        # 2. Load Benchmark Results (.csv)
        bench_file = os.path.join(exp_dir, "benchmark_results.csv")
        if os.path.exists(bench_file):
            try:
                # Read CSV directly
                df_b = pd.read_csv(bench_file)
                # Attach metadata to every row
                for k, v in meta.items():
                    df_b[k] = v
                all_bench_rows.append(df_b)
            except Exception as e:
                print(f"  ⚠️ Error loading {bench_file}: {e}")

    # Combine
    df_eval = pd.DataFrame(all_eval_rows)
    
    if all_bench_rows:
        df_benchmark = pd.concat(all_bench_rows, ignore_index=True)
    else:
        df_benchmark = pd.DataFrame()

    return df_eval, df_benchmark

def get_factors(df):
    """
    Identifies common vs unique factors across experiments.
    Returns: (common_dict, unique_keys)
    """
    # Potential keys to check
    keys = ["algo", "action_space", "traffic", "input", "reward", "curriculum", "fov_mask"]
    
    common = {}
    unique = []
    
    for k in keys:
        if k not in df.columns:
            continue
        # Drop NaNs
        vals = df[k].dropna().unique()
        if len(vals) == 1:
            common[k] = vals[0]
        elif len(vals) > 1:
            unique.append(k)
            
    return common, unique

def format_title_legend(df, common, unique):
    """
    Creates a title string from common factors (two lines if long),
    and a lambda function for labels from unique factors (exp-id-abbrev).
    """
    # 1. Build Title
    title_parts = [f"{k.replace('_', ' ').title()}: {v}" for k, v in common.items()]
    
    if len(title_parts) > 3:
        # Split into two lines
        mid = len(title_parts) // 2 + 1
        line1 = " | ".join(title_parts[:mid])
        line2 = " | ".join(title_parts[mid:])
        title_str = f"{line1}\n{line2}"
    else:
        title_str = " | ".join(title_parts) if title_parts else "All Experiments"
    
    # 2. Build Label Function with Abbreviations
    def get_label(row):
        # Format: exp-{id}-{abbrev_factor1}-{abbrev_factor2}...
        parts = [f"exp-{row['exp_id']}"]
        
        for k in unique:
            val = str(row[k]).lower()
            
            # Application specific abbreviations
            if k == "action_space":
                if "continuous" in val: val = "cont"
                elif "discrete" in val: val = "disc"
            elif k == "fov_mask":
                if "mask_off" in val or "mask_0" in val: val = "mask0"
                elif "mask_on" in val or "mask_1" in val: val = "mask1"
            elif k == "traffic":
                if "on" in val: val = "trf1"
                elif "off" in val: val = "trf0"
            elif k == "input":
                if "rgb" in val: val = "rgb"
                elif "masks" in val: val = "mask" # confusion risk with fov_mask? 
                # user example: exp-21-discrete-mask_off -> exp-21-disc-mask0
                # user requested: exp-id-cont-mask_0
                
            parts.append(val)
            
        return "-".join(parts)

    return title_str, get_label

# ===========================
# 📊 Plotting
# ===========================

def plot_success_rates(df_eval, save_dir):
    """
    Bar plot of Success Rate per Experiment.
    Includes legend with unique factors.
    """
    if df_eval.empty:
        print("⚠️ No evaluation data to plot.")
        return

    # Sort by ID
    df_eval = df_eval.sort_values("exp_id")
    
    # Factor Analysis
    common, unique = get_factors(df_eval)
    title_suffix, get_label = format_title_legend(df_eval, common, unique)

    plt.figure(figsize=(12, 8))
    
    # Create labels
    df_eval["label"] = df_eval.apply(get_label, axis=1)

    colors = sns.color_palette("muted", len(df_eval))
    ax = sns.barplot(
        data=df_eval, 
        x="exp_id", 
        y="success_rate", 
        hue="label",  # Use full label for legend
        palette=colors, 
        dodge=False
    )

    plt.title(f"Final Success Rate Comparison\n{title_suffix}", fontweight="bold", pad=20)
    plt.xlabel("Experiment ID", labelpad=10)
    plt.ylabel("Success Rate", labelpad=10)
    plt.ylim(0, 1.05)
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    # Annotate bars
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}', 
                        (p.get_x() + p.get_width() / 2., height), 
                        ha='center', va='bottom', 
                        fontsize=12, fontweight='bold', color='black')
            
    # Move legend outside
    plt.legend(title="Experiments", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "success_rates.png")
    plt.savefig(save_path, dpi=600)
    print(f"✅ Saved plot: {save_path}")
    plt.close()

def plot_training_efficiency(df_bench, save_dir):
    """
    Line plot: Global Step vs Success Threshold.
    Uses PchipInterpolator for monotonic smoothing.
    """
    if df_bench.empty:
        print("⚠️ No benchmark data to plot.")
        return

    # Factor Analysis
    common, unique = get_factors(df_bench)
    title_suffix, get_label = format_title_legend(df_bench, common, unique)

    plt.figure(figsize=(12, 8))
    
    experiments = df_bench["experiment"].unique()
    palette = sns.color_palette("bright", len(experiments))
    
    # Sort experiments by ID for consistent legend order
    # Extract IDs and sort
    sorted_experiments = sorted(experiments, key=lambda x: parse_experiment_name(x)["exp_id"] if parse_experiment_name(x)["exp_id"] else 0)

    for i, exp_name in enumerate(sorted_experiments):
        sub = df_bench[df_bench["experiment"] == exp_name].sort_values(["iteration", "threshold"])
        
        # Get one row to determine label
        first_row = sub.iloc[0]
        label = get_label(first_row)
        
        # Handle duplicate steps: keep the highest threshold reached at that step
        sub = sub.drop_duplicates(subset="iteration", keep="last")
        
        # Data points
        x = sub["iteration"].values
        y = sub["threshold"].values
        
        # Add (0,0) point for better curve starting if not present
        if 0 not in x:
            x = np.insert(x, 0, 0)
            y = np.insert(y, 0, 0.0)
        
        color = palette[i % len(palette)]

        # Smoothing
        if len(x) > 3:
            try:
                # Create smooth x axis
                x_smooth = np.linspace(x.min(), x.max(), 300)
                # Monotonic Cubic Interpolation
                interpolator = PchipInterpolator(x, y)
                y_smooth = interpolator(x_smooth)
                
                # Plot smooth line
                plt.plot(x_smooth, y_smooth, linewidth=3, color=color, label=label, alpha=0.9)
                # Plot original points lightly
                plt.scatter(x, y, color=color, s=40, alpha=0.4)
            except Exception as e:
                print(f"⚠️ Could not smooth {label}: {e}")
                plt.plot(x, y, marker="o", linewidth=3, label=label, color=color, alpha=0.9)
        else:
            plt.plot(x, y, marker="o", linewidth=3, label=label, color=color, alpha=0.9)

    plt.title(f"Training Efficiency: Learning Speed\n{title_suffix}", fontweight="bold", pad=20)
    plt.xlabel("Iterations", labelpad=10)
    plt.ylabel("Success Rate Achieved", labelpad=10)
    plt.grid(True, linestyle="--", alpha=0.5)
    
    # Legend
    plt.legend(title="Experiments", bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, "training_efficiency.png")
    plt.savefig(save_path, dpi=600)
    print(f"✅ Saved plot: {save_path}")
    plt.close()

# ===========================
# 🚀 Main
# ===========================

if __name__ == "__main__":
    df_eval, df_bench = load_data(ROOT)
    
    print(f"loaded {len(df_eval)} eval records.")
    print(f"loaded {len(df_bench)} benchmark records.")

    if not df_eval.empty:
        plot_success_rates(df_eval, ROOT)
    
    if not df_bench.empty:
        plot_training_efficiency(df_bench, ROOT)
