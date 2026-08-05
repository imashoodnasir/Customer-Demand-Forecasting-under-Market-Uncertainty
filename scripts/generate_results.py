from pathlib import Path
from bayesian_retail.visualization.plots import save_bar_plot

def main():
    output=Path("results/figures")
    save_bar_plot(
        ["Bayesian","TFT","PatchTST"],
        [1.0,1.3,1.5],
        output/"overall_performance.png"
    )
    print("Figures generated")

if __name__=="__main__":
    main()
