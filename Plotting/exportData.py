import os
import sys
import argparse

try:
    from io import StringIO, BytesIO  # python 3
except:
    from StringIO import StringIO  # python 2

import struct
import json
import csv


def import_data(import_file):
    '''
    Import data from import_file.
    Expect to find fixed width row.
    :param import_file:
    :return:
    '''
    mask = '9s14s5s'
    data = []

    count = 1
    with open(import_file, 'r') as f:
        for line in f:
            # unpack line to tuple
            fields = struct.unpack_from(mask, bytes(line, 'utf-8'))
            # skip any whitespace for each field
            # pack everything in a list and add to full dataset
            if count < 100:
                data.append(list([field.strip().decode() for field in fields]))
                count += 1
            else:
                break
    return data


def write_data(data, export_format):
    '''
    Dispatch call to a specific transformer and return dataset
    Exception is xlsx where we have to save data in a file
    :param data:
    :param export_format:
    :return:
    '''
    if export_format == 'csv':
        return write_csv(data)
    elif export_format == 'json':
        return write_json(data)
    elif export_format == 'xlsx':
        return write_xlsx(data)
    else:
        raise Exception("Illegal format defined")


def write_csv(data):
    '''
    Transform data into csv. Return csv as string.
    :param data:
    :return:
    '''
    f = StringIO()
    writer = csv.writer(f)
    for row in data:
        writer.writerow(row)
    # Get the content of the file-like object
    return f.getvalue()


def write_json(data):
    '''
    Transform data into json. Very straightforward.
    :param data:
    :return:
    '''
    j = json.dumps(data)
    return j


def write_xlsx(data):
    '''
    Write data into xlsx file.
    :param data:
    :return:
    '''
    from xlwt import Workbook
    book = Workbook()
    sheet1 = book.add_sheet("Sheet 1")
    row = 0
    for line in data:
        col = 0
        for datum in line:
            print(datum)
            sheet1.write(row, col, datum)
            col += 1
        row += 1
        # We have hard limit here of 65535 rows
        # that we are able to save in spreadsheet
        if row > 65535:
            print(sys.stderr, "Hit limit of # rows in sheet (65535)")
            break

    # XLS is special case where we have to
    # save the file and just return 0
    f = BytesIO()
    book.save(f)
    return f.getvalue()


if __name__ == '__main__':
    # parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("import_file", help="Path to a fixed-width data file.")
    parser.add_argument("export_format", help="Export format: json, csv, xlsx.")
    args = parser.parse_args()

    if args.import_file is None:
        print(sys.stderr, "You must provide valid export file format")
        sys.exit(-1)

    # Verify that given path is accessible file
    if not os.path.isfile(args.import_file):
        print(sys.stderr, "Given path is not a file: %s" % args.import_file)
        sys.exit(-1)

    # read from formatted fixed-width file
    data = import_data(args.import_file)

    # export data to specified format
    # to make this Unix-like pipe-able
    # we just print to stdout
    print(write_data(data, args.export_format))
