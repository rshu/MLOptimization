import sys

filename = sys.argv[1]

with open(filename, 'rb') as hugefile:
    chunksize = 100
    readable = ''
    # if you want to stop after certain number of blocks
    # put condition in the while

    while hugefile:
        # if you want to start not from 1st byte
        # do a hugefile.seek(skipbytes) to skip
        # skipbytes of bytes from the file start
        # tell() method can be used to get the position of File Handle
        start = hugefile.tell()
        print("starting at:", start)

        file_block = ''  # holds chunk_size of lines
        for _ in range(start, start + chunksize):
            line = str(hugefile.readline())  # ??
            # print(line)
            file_block = file_block + line
            print('file_block', type(file_block), file_block)
        readable = readable + file_block

        # tell where are we in file
        # file IO is usually buffered so tell() will not be precise for
        # every read.
        stop = hugefile.tell()
        print('readable', type(readable), readable)
        print('reading bytes from %s to %s' % (start, stop))
        print('read bytes total:', len(readable))

        # if you want to pause read between chucks
        # uncomment following line
        # input()
