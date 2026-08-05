from pathlib import Path

def main():
    Path("results/tables").mkdir(parents=True,exist_ok=True)
    Path("results/tables/example.tex").write_text(
        "% Generated manuscript table"
    )

if __name__=="__main__":
    main()
