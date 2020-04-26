import time
import os
import sys

if len(sys.argv) != 2:
    print("Please specify filename to read")
    sys.exit(-1)

filename = sys.argv[1]

if not os.path.isfile(filename):
    print("Given file: \"%s\" is not a file " % filename)

with open(filename, 'r') as f:
    # move to the end of file
    filesize = os.stat(filename)[6]
    # Change the current file position to filesize, and return the rest of the line
    f.seek(filesize)

    # endlessly loop
    while True:
        # tell() method can be used to get the position of File Handle
        where = f.tell()
        # try reading a line
        line = f.readline()
        # if empty, go back
        if not line:
            time.sleep(1)
            f.seek(where)
        else:
            # , at the end prevents print to add newline, as readline()
            # already read that
            print(line)
