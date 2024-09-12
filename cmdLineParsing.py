import argparse

def parse_args():
    parser = argparse.ArgumentParser(description = "This is a demonstration program")

    # Here we add an argument to the parser, specifying the expected type, a help message, etc.
    # parser.add_argument('-llms', type = str, required = True, help = 'Please provide an llms')
    parser.add_argument('-bs', type = str, required = True, help = 'Please provide a batch size')

    return parser.parse_args()

def main():
    args = parse_args()

    # Now we can use the argument value in our program
    # print(f"The provided line is : {args.llms}")
    print(f"The provided line is : {args.bs}")

if __name__ == '__main__':
    main()

'''
For testing the program -> in the command line, use this code 🔽
>> python .\cmdLineParsing.py -llms hello!
>> python .\cmdLineParsing.py -bs 6 
'''