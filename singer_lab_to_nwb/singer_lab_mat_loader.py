from mat_conversion_utils import convert_mat_file_to_dict
from pathlib import Path


class SingerLabMatLoader:

    def __init__(self, subject, date):
        self.subject = subject
        self.date = date

    def run_conversion(self, filenames, recs, output_type):
        # get filename and rec inputs
        filenames = filenames if isinstance(filenames, list) else [filenames]  # convert to list so we can iterate over
        self.filenames = [Path(f) for f in filenames]  # convert filenames to Path if not already
        self.recs = recs

        # determine output data type and run related function
        if output_type == 'scipy':
            data = self.convert_to_scipy_obj()
        elif output_type == 'concat_array':
            data = self.convert_recs_to_array_obj()
        elif output_type == 'array':
            data = self.convert_to_array_obj()
        elif output_type == 'dict':
            data = self.convert_to_dict_obj()

        return data

    def convert_to_scipy_obj(self, filename=None, rec=None):
        # use default values if no arguments given
        filename = filename or self.filenames
        rec = rec or self.recs

        # load matlab file
        matin = convert_mat_file_to_dict(str(filename[0]))
        stem = filename[0].stem.strip(str(rec))  # strip digits related to recording

        try:  # catch for first file vs subsequent ones
            mat_obj = matin[stem][int(self.subject) - 1][int(self.date) - 1][int(rec) - 1]  # subtract because 0-based
        except TypeError:
            mat_obj = matin[stem][int(self.subject) - 1][int(self.date) - 1]

        return mat_obj

    def convert_recs_to_array_obj(self):
        array_obj = []
        for ind, file in enumerate(self.filenames):
            mat_obj = self.convert_to_scipy_obj([file], self.recs[ind])
            array_obj.extend(mat_obj.data)

        return array_obj

    def convert_to_array_obj(self):
        mat_obj = self.convert_to_scipy_obj()
        array_obj = mat_obj.data

        return array_obj

    def convert_to_dict_obj(self):
        # load matlab file
        matin = convert_mat_file_to_dict(str(self.filenames[0]))

        key = [k for k in matin.keys() if not k.startswith('__')]
        dict_obj = matin[key[0]]

        return dict_obj