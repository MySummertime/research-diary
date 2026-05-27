from dataclasses import dataclass

@dataclass
class MarketTestConfig:
    """Parameters for market correlation test (independent from main model config)."""
    N_samples: int = 3000
    noise_level: float = 0.05
    D_base: float = 100.0
    P_base: float = 500.0
    delta_D: float = 0.2
    delta_C: float = 0.3
    S_base: float = 100.0
    H_base: float = 50.0
    delta_S: float = 0.25
    delta_H: float = 0.4


sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from projects.rhmvsp.config import ColorPalette


def test_market_correlation():
    cfg = MarketTestConfig()
    np.random.seed(7)
    N_samples = cfg.N_samples
    xi = np.random.uniform(-1, 1, N_samples)
    noise_level = cfg.noise_level
    noise_D = np.random.normal(0, noise_level, N_samples)
    noise_P = np.random.normal(0, noise_level, N_samples)
    noise_S = np.random.normal(0, noise_level, N_samples)
    noise_H = np.random.normal(0, noise_level, N_samples)
    D_base = cfg.D_base
    P_base = cfg.P_base
    delta_D = cfg.delta_D
    delta_C = cfg.delta_C
    Demand = D_base * (1 + delta_D * xi) + D_base * noise_D
    Demand_Penalty = P_base * (1 + delta_C * xi) + P_base * noise_P
    S_base = cfg.S_base
    H_base = cfg.H_base
    delta_S = cfg.delta_S
    delta_H = cfg.delta_H
    Supply = S_base * (1 + delta_S * xi) + S_base * noise_S
    Holding_Penalty = H_base * (1 + delta_H * xi) + H_base * noise_H

    # 4. Statistical Tests
    results = []

    # Demand vs Demand Penalty
    corr_dp, p_val_corr_dp = stats.pearsonr(Demand, Demand_Penalty)
    slope_dp, intercept_dp, r_value_dp, p_value_reg_dp, std_err_dp = stats.linregress(Demand, Demand_Penalty)
    # F-test for regression (equivalent to t-test squared for single variable)
    f_stat_dp = (r_value_dp**2 * (N_samples - 2)) / (1 - r_value_dp**2)
    p_val_f_dp = stats.f.sf(f_stat_dp, 1, N_samples - 2)
    
    results.append({
        "Model": "Demand vs Penalty",
        "Pearson Corr (r)": f"{corr_dp:.4f}",
        "T-test p-value (Corr)": f"{p_val_corr_dp:.2e}",
        "F-test Stat": f"{f_stat_dp:.2f}",
        "F-test p-value (Reg)": f"{p_val_f_dp:.2e}",
        "Conclusion": "Strong Positive Correlation" if corr_dp > 0.5 and p_val_corr_dp < 0.05 else "Weak/No Correlation"
    })

    # Supply vs Holding Penalty
    corr_sh, p_val_corr_sh = stats.pearsonr(Supply, Holding_Penalty)
    slope_sh, intercept_sh, r_value_sh, p_value_reg_sh, std_err_sh = stats.linregress(Supply, Holding_Penalty)
    f_stat_sh = (r_value_sh**2 * (N_samples - 2)) / (1 - r_value_sh**2)
    p_val_f_sh = stats.f.sf(f_stat_sh, 1, N_samples - 2)

    results.append({
        "Model": "Supply vs Holding Penalty",
        "Pearson Corr (r)": f"{corr_sh:.4f}",
        "T-test p-value (Corr)": f"{p_val_corr_sh:.2e}",
        "F-test Stat": f"{f_stat_sh:.2f}",
        "F-test p-value (Reg)": f"{p_val_f_sh:.2e}",
        "Conclusion": "Strong Positive Correlation" if corr_sh > 0.5 and p_val_corr_sh < 0.05 else "Weak/No Correlation"
    })

    # Print results
    print("="*60)
    print("MARKET COUPLING CORRELATION EXPERIMENT RESULTS")
    print("="*60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    print("="*60)
    
    # Generate separate KDE plots for a professional academic look
    import matplotlib.pyplot as plt
    import seaborn as sns
    import matplotlib.colors as mcolors
    from scipy.stats import linregress
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
    plt.rcParams['mathtext.fontset'] = 'custom'
    plt.rcParams['mathtext.rm'] = 'Times New Roman'
    plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
    plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
    sns.set_theme(style="whitegrid", font="Times New Roman")
    
    # Custom colormaps based on official palette but with steeper gradient
    cmap_terracotta = mcolors.LinearSegmentedColormap.from_list("tc", ["#FFFFFF", "#F5D6C6", ColorPalette.TERRACOTTA, "#8A3D1E"])
    cmap_celadon = mcolors.LinearSegmentedColormap.from_list("ce", ["#FFFFFF", "#E1EFEA", ColorPalette.CELADON, "#5B8D83"])

    # 1. Demand vs Penalty
    fig, ax1 = plt.subplots(figsize=(7, 6))
    
    # Scatter Points
    ax1.scatter(Demand, Demand_Penalty, color=ColorPalette.TERRACOTTA, alpha=0.5, s=8, zorder=1, label="Observations")
    
    # Trend Line
    slope_dp, intercept_dp, r_value_dp, p_value_dp, std_err_dp = linregress(Demand, Demand_Penalty)
    x_vals_dp = np.linspace(min(Demand), max(Demand), 100)
    ax1.plot(x_vals_dp, intercept_dp + slope_dp * x_vals_dp, color='gray', linestyle='-', alpha=0.7, linewidth=2.5, zorder=2, label=f"Trend Line ($r$={r_value_dp:.2f})")
    
    ax1.set_xlabel("Market Demand (Tons)", fontsize=14, fontname='Times New Roman')
    ax1.set_ylabel("Penalty Cost (¥/Ton)", fontsize=14, fontname='Times New Roman')
    ax1.tick_params(axis='both', direction='in', labelsize=12)
    ax1.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 12}, frameon=False, facecolor='none')
    ax1.grid(True, linestyle='--', alpha=0.7)
    sns.despine(ax=ax1)
    
    plot_path1 = os.path.join(os.path.dirname(__file__), "..", "..", "results", "market_coupling_demand_penalty.png")
    os.makedirs(os.path.dirname(plot_path1), exist_ok=True)
    plt.tight_layout()
    fig.savefig(plot_path1, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Supply vs Holding Penalty
    fig, ax2 = plt.subplots(figsize=(7, 6))
    
    # Scatter Points
    ax2.scatter(Supply, Holding_Penalty, color=ColorPalette.CELADON, alpha=0.5, s=8, zorder=1, label="Observations")
    
    # Trend Line
    slope_sh, intercept_sh, r_value_sh, p_value_sh, std_err_sh = linregress(Supply, Holding_Penalty)
    x_vals_sh = np.linspace(min(Supply), max(Supply), 100)
    ax2.plot(x_vals_sh, intercept_sh + slope_sh * x_vals_sh, color='gray', linestyle='-', alpha=0.7, linewidth=2.5, zorder=2, label=f"Trend Line ($r$={r_value_sh:.2f})")
    
    ax2.set_xlabel("Origin Supply (Tons)", fontsize=14, fontname='Times New Roman')
    ax2.set_ylabel("Holding Penalty (¥/Ton)", fontsize=14, fontname='Times New Roman')
    ax2.tick_params(axis='both', direction='in', labelsize=12)
    ax2.legend(loc='upper left', prop={'family': 'Times New Roman', 'size': 12}, frameon=False, facecolor='none')
    ax2.grid(True, linestyle='--', alpha=0.7)
    sns.despine(ax=ax2)
    
    plot_path2 = os.path.join(os.path.dirname(__file__), "..", "..", "results", "market_coupling_supply_penalty.png")
    plt.tight_layout()
    fig.savefig(plot_path2, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved to:\n  - {os.path.abspath(plot_path1)}\n  - {os.path.abspath(plot_path2)}")

if __name__ == "__main__":
    test_market_correlation()
