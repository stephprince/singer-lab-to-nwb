import glob
import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from pynwb.epoch import TimeIntervals
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class

from mat_conversion_utils import convert_mat_file_to_dict
from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile

class SingerLabPreprocessingInterface(BaseDataInterface):
    """
    Data interface for preprocessed, filtered data in standard Singer Lab format

    Data is stored in
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        source_schema = dict(
            required=['processed_data_folder', 'channel_map_path'],
            properties=dict(
                file_path=dict(type='string'),
                channel_map_path=dict(type='string'),
            ),
        )
        return source_schema

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
        # TODO - figure out what all this schema setup does and if I need it
        metadata_schema["properties"]["Ecephys"] = get_base_schema(tag="Ecephys")
        metadata_schema["properties"]["Ecephys"]["required"] = ["Device", "ElectrodeGroup"]
        metadata_schema["properties"]["Ecephys"]["properties"] = dict(
            Device=dict(type="array", minItems=1, items={"$ref": "#/properties/Ecephys/properties/definitions/Device"}),
            ElectrodeGroup=dict(
                type="array", minItems=1, items={"$ref": "#/properties/Ecephys/properties/definitions/ElectrodeGroup"}
            ),
            Electrodes=dict(
                type="array",
                minItems=0,
                renderForm=False,
                items={"$ref": "#/properties/Ecephys/properties/definitions/Electrodes"},
            ),
        )
        # Schema definition for arrays
        metadata_schema["properties"]["Ecephys"]["properties"]["definitions"] = dict(
            Device=get_schema_from_hdmf_class(Device),
            ElectrodeGroup=get_schema_from_hdmf_class(ElectrodeGroup),
            Electrodes=dict(
                type="object",
                additionalProperties=False,
                required=["name"],
                properties=dict(
                    name=dict(type="string", description="name of this electrodes column"),
                    description=dict(type="string", description="description of this electrodes column"),
                ),
            ),
        )
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()

        # add ecephys info
        brain_regions = self.source_data['brain_regions']
        brain_region_groups = [br for br in brain_regions for _ in range(2)]  # double list so region for each shank
        channel_groups = range(len(brain_region_groups))

        spikegadgets = [dict(
            name="spikegadgets_mcu",
            description="Two NeuroNexus silicon probes with 2 (shanks) x 32 (channels) on each probe were inserted into"
                        " hippocampal CA1 and medial prefrontal cortex. Probes were 64-chan, poly5 Takahashi probe "
                        "formats. Electrophysiological data were acquired using a SpikeGadgets MCU system digitized "
                        "with 30 kHz rate. Analog and digital channels were acquired using the SpikeGadgets ECU system."
        ),
            dict(name="spikegadgets_ecu",
                 description="Analog and digital inputs of SpikeGadgets system. Max -10 to 10V for analog channels.")
        ]

        electrode_group = [dict(
            name=f'shank{n + 1}',
            description=f'shank{n + 1} of NeuroNexus probes. Shanks 1-2 belong to probe 1, Shanks 3-4 belong to probe 2',
            location=brain_region_groups[n],
            device='spikegadgets_mcu')
            for n, _ in enumerate(channel_groups)
        ]
        electrode_group.append(dict(
            name='analog_inputs',
            description='analog inputs to SpikeGadgets system. Channel IDs are unique to the project and task.',
            location='none',
            device='spikegadgets_ecu')
        )

        metadata["Ecephys"] = dict(
            Device=spikegadgets,
            ElectrodeGroup=electrode_group,
            # Electrodes=[
            #     dict(name="electrode_number",
            #          description="0-indexed channel within the probe. Channels 0-31 belong to shank 1 and channels 32-64"
            #                      " belong to shank 2"),
            #     dict(name="group_name", description="Name of the electrode group this electrode is part of")
            # ]
        )

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # get general session details from first channel recording file
        processed_data_folder = Path(self.source_data['processed_data_folder'])
        subject_num = metadata['Subject']['subject_id'][1:]
        session_date = processed_data_folder.stem.split(f'{subject_num}_')[1]
        brain_regions = np.unique([e['location'] for e in metadata['Ecephys']['ElectrodeGroup']])

        # extract recording times to epochs, same for both brain regions
        base_path = processed_data_folder / brain_regions[0]
        filenames = glob.glob(str(base_path) + '/0/eeg*.mat')  # use eeg files bc most consistent name across singer lab

        recording_epochs = get_recording_epochs(filenames, self.source_data['vr_files'], subject_num, session_date)
        nwbfile.add_time_intervals(recording_epochs)

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



def get_recording_epochs(filenames, vr_files, subject_num, session_date):
    # make time intervals structure
    recording_epochs = TimeIntervals(
        name="recording_epochs",
        description='During one recording session, the acquisition is stopped and restarted for switching '
                    'behavioral periods, refilling with saline, etc. Electrophysiological files are stitched '
                    'together and these epochs indicate the start and stop times within a recording session. These '
                    'epochs are NOT continuous and there is a short interval of approx 1-5 min between them.'
    )
    recording_epochs.add_column(name="behavior_task",
                                description="indicates whether VR task was displayed on the screen (1), or if the "
                                            "screen was left covered/blank as a baseline period (0)")

    # loop through files to get time intervals
    start_time = 0.0  # initialize start time to 0 sec
    for ind, f in enumerate(filenames):
        # load singer mat structure
        eeg_mat = convert_singer_mat_to_scipy_obj(f, subject_num, session_date, ind)

        # add to nwb file
        duration = (len(eeg_mat.time) / eeg_mat.samprate)  # in seconds
        recording_epochs.add_row(start_time=start_time, stop_time=start_time + duration,
                                 behavior_task=vr_files[ind])
        start_time = start_time + duration

    return recording_epochs

def get_analog_timeseries(data_folder, subject_num, session_date):
    # define analog signals saved in data
    analog_signals = ['licks', 'rotVelocity', 'transVelocity']
    analog_signals_descript = ['signal from photointerrupter circuit, values of 5V indicate the mouse tongue is has '
                               'broken through the laser beam, values of 0V indicate baseline',
                               'movement of the spherical treadmill along the roll axis as measured by an optical'
                               'gaming mouse and filtered through a lab view function, maximum values of -10 and 10V',
                               'movement of the spherical treadmill along the pitch axis as measured by an optical'
                               'gaming mouse and filtered through a lab view function, maximum values of -10 and 10V']

    # look through signals and generate electrical series
    analog_dict = dict(zip(analog_signals, analog_signals_descript))
    for name, descript in analog_dict.items():
        # append the electrical series across recording files
        filenames = glob.glob(str(data_folder) + f'/{name}*.mat')
        for ind, f in enumerate(filenames):
            # load singer lab mat structure
            analog_obj = convert_singer_mat_to_scipy_obj(f, subject_num, session_date, ind)

            # append series data
            full_series[name].append(analog_obj.data)
            rate[name] = analog_obj.sampling_rate

        # general electrical series objects
        analog_obj[name] = ElectricalSeries(name=name,
                                           data=H5DataIO(full_series[name], compression='gzip'),
                                           starting_time=0,
                                           rate=rate[name],
                                           description=descript)
    return analog_obj

def convert_singer_mat_to_scipy_obj(filename, subject_num, session_date, rec_num):
    # load matlab file
    matin = convert_mat_file_to_dict(filename)

    try:  # catch for first file vs subsequent ones
        mat_obj = matin['eeg'][int(subject_num) - 1][int(session_date) - 1][rec_num]  # subtract 1 because 0-based indexing
    except TypeError:
        mat_obj = matin['eeg'][int(subject_num) - 1][int(session_date) - 1]

    return mat_obj

