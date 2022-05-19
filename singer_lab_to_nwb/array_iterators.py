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
            for r in self.recs:
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


class MultiDimMultiFileArrayIterator(MultiFileArrayIterator):

    def __init__(self, channel_dirs, dim_names, mat_loader, recs, num_samples, metric_row):
        """
        :param channel_dirs: List of dirs with channel specific data
        :param num_steps: Number of timesteps per channel
        :return:
        """
        self.shape = (num_samples, len(channel_dirs), len(dim_names))
        self.channel_dirs = channel_dirs
        self.dim_names = dim_names
        self.mat_loader = mat_loader
        self.recs = recs
        self.metric_row = metric_row  # which row to use (e.g. phase, amplitude, or envelope)
        self.num_chunks = len(channel_dirs)*len(dim_names)
        self.__curr_chunk = 0

    def __next__(self):
        """
        Return in each iteration the data from a single file
        """
        if self.__curr_chunk < self.num_chunks:
            # find which dimension we're currently iterating through and get relevant filenames
            dim = int(np.floor(self.__curr_chunk/len(self.channel_dirs)))  # e.g., less than 64 chunks -> dim 0
            ch = self.__curr_chunk - dim*len(self.channel_dirs)
            filenames = []
            for r in self.recs:
                filenames.append(self.channel_dirs[ch] / f'{self.dim_names[dim]}{r}.mat')

            # load and concatenate data across recording files
            temp_data = self.mat_loader.run_conversion(filenames, self.recs, 'concat_array')
            curr_data = np.array(temp_data)[:, self.metric_row]

            # create data chunk
            self.__curr_chunk += 1
            return DataChunk(data=np.squeeze(curr_data),
                             selection=np.s_[:, ch, dim])
        else:
            raise StopIteration

    next = __next__

class MultiFileSpikeGadgetsIterator(MultiFileArrayIterator):

    def __init__(self, filenames, data_loader, recs, num_channels, num_samples_per_rec):
        """
        :param channel_dirs: List of dirs with channel specific data
        :param num_steps: Number of timesteps per channel
        :return:
        """
        self.shape = (np.sum(num_samples_per_rec), num_channels)
        self.num_channels = num_channels
        self.filenames = filenames
        self.data_loader = data_loader
        self.recs = recs
        self.num_samples_per_rec = num_samples_per_rec
        self.num_chunks = len(filenames)
        self.__curr_chunk = 0

    def __next__(self):
        """
        Return in each iteration the data from a single file
        """
        if self.__curr_chunk < self.num_chunks:

            # get info about current chunk indexing/filename
            rec_num = int(np.floor(self.__curr_chunk/self.num_channels))  # e.g., less than 64 chunks -> dim 0
            ch = self.__curr_chunk - rec_num*self.num_channels
            start_chunk_ind = int(0+np.sum(self.num_samples_per_rec[:rec_num]))
            end_chunk_ind = int(np.sum(self.num_samples_per_rec[:rec_num+1]))

            # load data
            temp_data = self.data_loader(self.filenames[self.__curr_chunk])
            curr_data = temp_data['data'].astype('int64')

            # create data chunk
            self.__curr_chunk += 1
            return DataChunk(data=curr_data,
                             selection=np.s_[start_chunk_ind:end_chunk_ind, ch])
        else:
            raise StopIteration

    next = __next__

    @property
    def dtype(self):
        return np.dtype('int64')
