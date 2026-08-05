import argparse

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",default="full")
    args=parser.parse_args()

    print("Running reproduction pipeline:", args.mode)
    print("1. Data preparation")
    print("2. Feature generation")
    print("3. Model training")
    print("4. Evaluation")
    print("5. Tables and figures")

if __name__=="__main__":
    main()
