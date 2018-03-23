
# read a file, break to columns
# returns total headers, the continuous numeric headers, and the symbolic (binary or string) headers
def make_col(path):
    with open(path) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    headers = []
    continuous = []
    symbolic = []
    for name in content:
        names = name.split(":")
        if len(names) > 1:
            header = names[0]
            headers.append(header)
            # print names[1]
            if names[1] == " continuous.":
                continuous.append(header)
            if names[1] == " symbolic.":
                symbolic.append(header)
    headers.append("class")
    return headers, continuous, symbolic

if __name__ == "__main__":
    headers, continuous, symbolic = make_col("data/kddcup.names")
    print headers
    print continuous
    print symbolic