# Read everything in the data file into lines
# lines is a list, each element is one line in the data file
def readLDFdata(filepath):
    file_path = filepath
    file = open("" + file_path + ".txt", "r")
    lines = file.readlines()


    # Finds where the raw data starts
    # Skips the two blank lines at very beginning
    i = 0
    while lines[i] != "4) Trace Data\n":
        i += 1
    i += 2


    # Read All Raw LDF Data
    raw_data_ldf = []
    for j in range(i, len(lines)):
        line = lines[j]
        ldf_val = float(line.split("\t")[4])
        raw_data_ldf.append(ldf_val)

    # See if the last datapoint is correct
    return raw_data_ldf