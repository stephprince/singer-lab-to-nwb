import glob
import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class
from scipy.io import loadmat

from mat_conversion_utils import convert_mat_file_to_dict


class SingerLabPreprocessingInterface(BaseDataInterface):
    """
    Data interface for preprocessed, filtered data in standard Singer Lab format

    Data is stored in
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        source_schema = dict(
            required=['processed_data_folder','channel_map_path'],
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
            name="spikegadgets",
            description="Two NeuroNexus silicon probes with 2 (shanks) x 32 (channels) on each probe were inserted into"
                        " hippocampal CA1 and medial prefrontal cortex. Probes were 64-chan, poly5 Takahashi probe "
                        "formats. Electrophysiological data were acquired using a SpikeGadgets system digitized with 30"
                        " kHz rate."
        )
        ]

        electrode_group = [dict(
            name=f'shank{n + 1}',
            description=f'shank{n + 1} of NeuroNexus probes. Shanks 1-2 belong to probe 1, Shanks 3-4 belong to probe 2',
            location=brain_region_groups[n],
            device='spikegadgets')
            for n, _ in enumerate(channel_groups)
        ]

        metadata["Ecephys"] = dict(
            Device=spikegadgets,
            ElectrodeGroup=electrode_group,
            Electrodes=[
                dict(name="electrode_number",
                     description="0-indexed channel within the probe. Channels 0-31 belong to shank 1 and channels 32-64"
                                 " belong to shank 2"),
                dict(name="group_name", description="Name of the electrode group this electrode is part of")
            ]
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
        # use eeg files as basis bc likely to be most consistent across singer lab recordings
        base_path = processed_data_folder / brain_regions[0] / '0'
        filenames = glob.glob(str(base_path) + '/eeg*.mat')
        recording_epochs = TimeIntervals(
            name="recording_epochs",
            description='During one recording session, the acquisition is stopped and restarted for switching '
                        'behavioral periods, refilling with saline, etc. Electrophysiological files are stitched '
                        'together and these epochs indicate the start and stop times within a recording session. These '
                        'files are NOT continuous and there is a short interval of approx 1-5 min between them.'
        )
        recording_epochs.add_column(name="behavior_task",
                                    description="indicates whether VR task was displayed on the screen (1), or if the "
                                                "screen was left covered/blank as a baseline period (0)")

        start_time = 0.0  # initialize start time to 0 sec
        for ind, f in enumerate(filenames):
            # determine recording duration
            matin = convert_mat_file_to_dict(f)
            try:  # catch for first file vs subsequent ones
                eeg_mat = matin['eeg'][int(subject_num) - 1][int(session_date) - 1][ind] # subtract 1 because 0-based indexing
            except TypeError:
                eeg_mat = matin['eeg'][int(subject_num) - 1][int(session_date) - 1]

            # add to nwb file
            duration = (len(eeg_mat.time) / eeg_mat.samprate) # in seconds
            recording_epochs.add_row(start_time=start_time, stop_time=start_time+duration, recording_epochs=temp)
            start_time = start_time + duration

        nwbfile.add_time_intervals(recording_epochs)

        #
        for region in brain_regions:
            mat = matin['eeg'][subject_num - 1][session_date - 1] # subtract 1 because 0-based indexing


        # use the first channel of the first region for this

        # extract analog signals
        analog_signals = ['licks', 'rotVelocity', 'transVelocity'] # stored as acquisition time series?
        digital_signals = ['delay', 'trial', 'update'] # stored as behavioral events

        # extract digital signals

        # load probe and channel details
        probe_file_path = Path(self.source_data['channel_map_path'])
        df = pd.read_csv(probe_file_path)

        # extract filtered lfp signals



        # if Path(mat_file).is_file():
        #     # convert the mat file into dict and data frame format
        #     matin = convert_mat_file_to_dict(mat_file)

