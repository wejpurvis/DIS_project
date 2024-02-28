"""
Reads .cel files for python manipulation
Specifically developed for the CEL files used in this project
"""

try:
    import numpy as np
except ImportError:
    print("Install numpy to use this tool")


class Record:
    def __init__(self):
        self.NumberCells = None
        self.intensities = None
        self.stdevs = None
        self.pixels = None
        self.nrows = None
        self.ncols = None
        self.nmask = None
        self.mask = None
        self.noutliers = None
        self.outliers = None
        self.modified = None
        self.nmodified = None


def parse_cel_file(file_path):

    record = Record()

    # open the .cel file
    with open(file_path, "rb") as f:
        content = f.readlines()

    # extract data
    section = None
    for idx, line in enumerate(content):
        line = line.decode("utf-8").strip()

        if not line:
            continue
        # set current section
        if line == "[HEADER]":
            section = line
        elif line == "[INTENSITY]":
            section = line
        elif line == "[MASKS]":
            section = line
        elif line == "[OUTLIERS]":
            section = line
        elif line == "[MODIFIED]":
            section = line

        # parse current section
        else:
            # get columns and rows from header
            if section == "[HEADER]":
                key, value = line.split("=", 1)
                if key == "Cols":
                    record.ncols = int(value)
                elif key == "Rows":
                    record.nrows = int(value)

                if record.ncols and record.nrows:
                    record.intensities = np.zeros((record.nrows, record.ncols))
                    record.stdevs = np.zeros((record.nrows, record.ncols))
                    record.pixels = np.zeros((record.nrows, record.ncols))
                    record.mask = np.zeros((record.nrows, record.ncols), bool)
                    record.outliers = np.zeros((record.nrows, record.ncols), bool)
                    record.modified = np.zeros((record.nrows, record.ncols))

            # get intensities, stdevs and pixels
            elif section == "[INTENSITY]":

                if line[0] == "N":
                    key, num_cells = line.split("=", 1)
                    record.NumberCells = int(num_cells)
                    if record.NumberCells != record.ncols * record.nrows:
                        raise ValueError(
                            "Number of cells does not match the number of rows and columns"
                        )
                elif line[0] == "C":
                    continue
                else:
                    parts = line.split()
                    y = int(parts[0])
                    x = int(parts[1])
                    record.intensities[y, x] = float(parts[2])
                    record.stdevs[y, x] = float(parts[3])
                    record.pixels[y, x] = int(parts[4])

            elif section == "[MASKS]":
                if line[0] == "N":
                    key, num_masks = line.split("=", 1)
                    record.nmask = int(num_masks)
                elif line[0] == "C":
                    continue
                else:
                    parts = line.split()
                    y = int(parts[0])
                    x = int(parts[1])
                    record.mask[y, x] = True

            elif section == "[OUTLIERS]":
                if line[0] == "N":
                    key, num_outliers = line.split("=", 1)
                    record.noutliers = int(num_outliers)
                elif line[0] == "C":
                    continue
                else:
                    parts = line.split()
                    y = int(parts[0])
                    x = int(parts[1])
                    record.outliers[y, x] = True

            elif section == "[MODIFIED]":
                if line[0] == "N":
                    key, num_modified = line.split("=", 1)
                    record.nmodified = int(num_modified)
                elif line[0] == "C":
                    continue
                else:
                    parts = line.split()
                    y = int(parts[0])
                    x = int(parts[1])
                    record.modified[y, x] = int(parts[2])

    return record
