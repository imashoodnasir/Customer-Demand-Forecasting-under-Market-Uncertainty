import pandas as pd
from bayesian_retail.results.export import export_latex_table

def main():
    df=pd.DataFrame({
        "Method":["Bayesian","TFT","PatchTST"],
        "RMSE":[1.9,2.3,2.5]
    })
    export_latex_table(
        df,
        "results/tables/overall_performance.tex"
    )

if __name__=="__main__":
    main()
