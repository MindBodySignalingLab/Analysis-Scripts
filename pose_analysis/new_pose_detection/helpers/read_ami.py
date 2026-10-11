def readAMIdata(filepath):
    file_path = filepath
    file = open(file_path + ".csv", "r")
    lines = file.readlines()

    # Finds where the raw data starts
    # Skips the one blank line at very beginning
    i = 0
    while lines[i] != "No,Time (hr:min:sec),LeftLU,Event messages\n":
        i += 1
    i += 1

    # Read how many lines we need to read in AMI
    numDataLines = int(lines[13].split(",")[1])

    raw_data_ami = []
    for j in range(i, i + numDataLines):
        line = lines[j]
        ami_val = float(line.split(",")[2])
        raw_data_ami.append(ami_val)

    # See if the last datapoint is correct
    return raw_data_ami