import numpy as np

from hdmf.data_utils import AbstractDataChunkIterator, DataChunk  # noqa: F811

class MultiDimMultiFileArrayIterator(AbstractDataChunkIterator):

    def __init__(self, channel_dirs, dim_names, mat_loader, recs, num_samples, metric_row):
        """
        :param channel_dirs: List of dirs with channel specific data
        :param num_steps: Number of timesteps per channel
        :return:
        """
        self.shape = (num_samples, len(channel_dirs), len(filenames))
        self.channel_dirs = channel_dirs
        self.dim_names = dim_names
        self.mat_loader = mat_loader
        self.recs = recs
        self.metric_row = metric_row  # which row to use (e.g. phase, amplitude, or envelope)
        self.num_chunks = len(channel_dirs)*len(filenames)
        self.__curr_chunk = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return in each iteration the data from a single file
        """
        if self.__curr_chunk < len(self.num_chunks):
            # find which dimension we're currently iterating through and get relevant filenames
            dim = np.ceil(self.__curr_chunk/len(channel_dirs))  # e.g., less than 64 chunks -> dim 0
            filenames = []
            for r in self.recs:  # TODO - see if faster to establish chunks as separate files or extend here
                filenames.append(self.channel_dirs[self.__curr_index] / f'{self.dim_names[dim]}{r}.mat')

            # load and concatenate data across recording files
            temp_data = self.mat_loader.run_conversion(filenames, self.recs, 'concat_array')
            curr_data = np.array(temp_data)[:, self.metric_row]

            # create data chunk
            self.__curr_chunk += 1
            return DataChunk(data=np.array(curr_data),
                             selection=np.s_[:, ch, dim])
        else:
            raise StopIteration

    next = __next__

    def recommended_chunk_shape(self):
        return None   # Use autochunking

    def recommended_data_shape(self):
        return self.shape

    @property
    def dtype(self):
        return np.dtype('float64')

    @property
    def maxshape(self):
        return self.shape
