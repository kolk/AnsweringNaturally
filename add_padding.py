import argparse

def add_padding(input_file, output_file):
    with open(input_file, "r") as f, open(output_file, "w") as out_f:
        for line in f:
            l= len(line.strip().split(' '))
            padding = ""
            num_of_pads = 50 - l - 1
            for i in range(num_of_pads):
                    padding += "<blank> "
            out_line = " ".join(line.strip().split(' ')) + " </s> " + padding + "\n"
            out_f.write(out_line)

parser = argparse.ArgumentParser()
parser.add_argument("-input", "--input", help="Input file")
parser.add_argument("-output", "--output", help="Output file")
args = parser.parse_args()
add_padding(args.input, args.output)

