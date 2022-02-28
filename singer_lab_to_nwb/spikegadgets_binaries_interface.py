import glob
import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, ElectricalSeries
from pynwb.epoch import TimeIntervals
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class

from mat_conversion_utils import convert_mat_file_to_dict
from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
from spikeinterface.extractors import SpikeGadgetsRecordingExtractor
from nwb_conversion_tools.utils.spike_interface import add_electrical_series

class SpikeGadgetsBinariesInterface(BaseDataInterface):
    """
    Data interface for preprocessed, filtered data in standard Singer Lab format

    Data is stored in
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        source_schema = dict(
            required=['raw_data_folder', 'brain_regions'],
            properties=dict(
                file_path=dict(type='string'),
                channel_map_path=dict(type='string'),
            ),
        )
        return source_schema

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()

        return metadata_schema


    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # get general session details
        raw_data_folder = Path(self.source_data['raw_data_folder'])
        subject_num = metadata['Subject']['subject_id'][1:]
        session_date = raw_data_folder.stem.split(f'{subject_num}_')[1]

        # create electrode groups for analog and  digital data
        device = nwbfile.create_device(name=metadata['Ecephys']['Device'][1]['name'])  # TODO - search for ECU device
        electrode_group = metadata['Ecephys']['ElectrodeGroup'][-1]
        nwbfile.create_electrode_group(name=electrode_group['name'],
                                       description=electrode_group['description'],
                                       location=electrode_group['location'],
                                       device=device)

        # extract analog signals (continuous, stored as acquisition time series)
        analog_obj = get_analog_timeseries(raw_data_folder, nwbfile)

        for analog_ts in analog_obj.values():
            nwbfile.add_acquisition(analog_ts)

        # extract digital signals (non-continuous, stored as behavioral events)
        digital_signals = ['delay', 'trial', 'update']

        # load probe and channel details
        probe_file_path = Path(self.source_data['channel_map_path'])
        df = pd.read_csv(probe_file_path)

        # extract filtered lfp signals
        signal_bands = ['eeg', 'theta', 'nontheta', 'ripple']
        for br in brain_regions:
            for ch in channels:
                base_path = processed_data_folder / br / ch


        # extract lfp event times
        event_types

def get_analog_timeseries(data_folder, nwbfile):
    # define analog signals saved in data
    analog_signals = ['licks', 'rotVelocity', 'transVelocity']
    analog_signals_descript = ['signal from photointerrupter circuit, values of 5V indicate the mouse tongue is has '
                               'broken through the laser beam, values of 0V indicate baseline',
                               'movement of the spherical treadmill along the roll axis as measured by an optical'
                               'gaming mouse and filtered through a lab view function, maximum values of -10 and 10V',
                               'movement of the spherical treadmill along the pitch axis as measured by an optical'
                               'gaming mouse and filtered through a lab view function, maximum values of -10 and 10V']
    analog_descript_dict = dict(zip(analog_signals, analog_signals_descript))
    analog_signals_chan_ID = [1, 2, 4]  # TODO - have this as a project input
    analog_chan_dict = dict(zip(analog_signals, analog_signals_chan_ID))

    # look through signals and generate electrical series
    analog_obj = dict.fromkeys(analog_signals)
    for name, chan in analog_chan_dict.items():

        # concatenate analog signal across recordings
        analog_filenames = list(data_folder.glob(f'**/*.analog_ECU_Ain{chan}.dat'))
        full_signal = []
        for file in analog_filenames:  # TODO - only add recordings based on the ephys spreadsheet include/exclude info
            analog_data = readTrodesExtractedDataFile(file)
            full_signal.extend(analog_data['data'])

        # generate electrodes and electrode region
        nwbfile.add_electrode(id=chan+128,  # TODO - fix hardcoding
                              x=np.nan, y=np.nan, z=np.nan,
                              imp=np.nan,
                              location='none',
                              filtering='none',
                              group=nwbfile.electrode_groups['analog_inputs'])
        elec_table_region = nwbfile.create_electrode_table_region(region=[0], description=str(analog_data['id']))

        # general electrical series objects
        analog_obj[name] = ElectricalSeries(name=name,
                                            data=H5DataIO(full_signal, compression='gzip'),
                                            starting_time=0.0,
                                            rate=float(analog_data['clockrate']),   # should be the same across files so read from last
                                            electrodes=elec_table_region,
                                            description=analog_descript_dict[name],   # should be the same across files so read from last
                                            comments=f"Channel ID was {analog_data['id']}")
    return analog_obj

def get_digital_events(data_folder):
    digital_signals_chan_ID = [1, 2, 3, 4]  # TODO - have this as a project input
    digital_signals = ['sync', 'trial', 'update', 'delay']


