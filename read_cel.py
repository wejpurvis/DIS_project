"""
Reads .cel files for python manipulation
Specifically developed for the CEL files used in this project
"""

try:
    import numpy as np
except ImportError:
    print("Install numpy to use this tool")


class Record:
    """
    Stores information in a cel file.

    Attributes:
        NumberCells (int): The number of cells.
        intensities (list): The intensities of the cells.
        stdevs (list): The standard deviations of the cells.
        pixels (list): The pixel values of the cells.
        nrows (int): The number of rows in the record.
        ncols (int): The number of columns in the record.
        nmask (int): The number of masked cells.
        mask (list): The masked cells.
        noutliers (int): The number of outlier cells.
        outliers (list): The outlier cells.
        modified (bool): Indicates if the record has been modified.
        nmodified (int): The number of modified cells.
        raw_content (list): The raw content of the .cel file.
    """

    def __init__(self):
        """
        Initializes the attributes of the class.
        """
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
        self.raw_content = []

    def __str__(self):
        """
        Returns a string representation of the class.

        Returns
        -------
        str
            A string representation of the class.
        """
        return "".join(line.decode("utf-8") for line in self.raw_content)


def parse_cel_file(file_path):
    """
    Parses a .cel file and populates a Record instance with its content.

    Parameters
    ----------
    file_path : str
                The path to the .cel file to be parsed.

    Returns
    -------
    Record : object
        An instance of the Record class populated with the .cel file's data.

    Raises
    ------
    ValueError
        If the number of cells does not match the expected number based on nrows and ncols.
    ImportError
        If numpy is not installed.

    Examples
    --------
    >>> record = parse_cel_file("path/to/file.cel")
    >>> print(record.ncols) # prints the number of columns in the record
    >>> print(record.nrows) # prints the number of rows in the record
    >>> print(record.intensities) # prints the intensities of the cells
    """
    record = Record()

    # open the .cel file
    with open(file_path, "rb") as f:
        content = f.readlines()
        record.raw_content = content

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
