import argparse
import pandas as pd
import json

def main(input, output=None):
    if output == None:
        output = "_CORA.".join(input.split("."))

    poleval_data = pd.read_csv(input, sep="\t", header=None)
    poleval_data["id"] = poleval_data.index
    poleval_data["lang"] = "pl"
    poleval_data = poleval_data.rename(columns={0: "question"})
    poleval_dicts = poleval_data.to_dict('records')

    print(f"Saving converted file to {output}")

    with open(output, 'w') as f:
        f.write('\n'.join(map(json.dumps, poleval_dicts)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output", help="Path to output file", required=False)
    parser.add_argument("--input", help="Path to CORA file", required=True)

    args = parser.parse_args()
    main(**vars(args))