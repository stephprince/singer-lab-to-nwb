import numpy as np

from hdmf.data_utils import AbstractDataChunkIterator, DataChunk  # noqa: F811

class MultiFileArrayIterator(AbstractDataChunkIterator):

    def __init__(self, channel_dirs, filename, mat_loader, recs, num_steps, ):
        """
        :param channel_dirs: List of dirs with channel specific data
        :param num_steps: Number of timesteps per channel
        :return:
        """
        self.shape = (num_steps, len(channel_dirs))
        self.channel_dirs = channel_dirs
        self.filename = filename
        self.mat_loader = mat_loader
        self.recs = recs
        self.num_steps = num_steps
        self.__curr_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        """
        Return in each iteration the data from a single file
        """
        if self.__curr_index < len(self.channel_dirs):
            # get list of filenames to use
            filenames = []
            for r in self.recs:  # TODO - see if faster to establish chunks as separate files or extend here
                filenames.append(self.channel_dirs[self.__curr_index] / f'{self.filename}{r}.mat')

            # load and concatenate data across recording files
            curr_data = self.mat_loader.run_conversion(filenames, self.recs, 'concat_array')

            # create data chunk
            i = self.__curr_index
            self.__curr_index += 1
            return DataChunk(data=np.array(curr_data),
                             selection=np.s_[:, i])
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
