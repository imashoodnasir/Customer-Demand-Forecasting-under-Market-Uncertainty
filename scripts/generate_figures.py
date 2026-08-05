from pathlib import Path
import matplotlib.pyplot as plt

def main():
    Path("results/figures").mkdir(parents=True,exist_ok=True)
    plt.figure()
    plt.plot([1,2,3],[1,4,9])
    plt.savefig("results/figures/example.png")
    plt.close()

if __name__=="__main__":
    main()
