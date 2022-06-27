import argparse


def convert_line(line: str) -> str:
    # if line[1] == '"':
    #     ...
    # elif line[1] == "'":
    #     ...
    # else:
    #     print("Unexpected quote:", line)
    #     exit()
    line = line.strip()
    val: str = eval(line).decode("utf-8")
    val = val.replace("<extra_id_0>", "")
    return val


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert model output to geval-readable format"
    )
    parser.add_argument("input", metavar="input.txt", type=str, help="Input file")
    parser.add_argument("output", metavar="output.txt", type=str, help="Output file")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        lines = list(map(convert_line, f.readlines()))

    # for line in lines:
    #     print(line)

    with open(args.output, "w") as f:
        f.write("\n".join(lines))
