import glob
import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, ElectricalSeries, LFP, FilteredEphys
from pynwb.epoch import TimeIntervals
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class

from multi_file_array_iterator import MultiFileArrayIterator
from singer_lab_mat_loader import SingerLabMatLoader
from update_task_conversion_utils import check_module

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
        channel_groups = range(len(brain_regions))

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
            name=f'probe{n + 1}',
            description=f'probe{n + 1} of NeuroNexus probes.  Channels 0-31 belong to shank 1 and channels 32-64 '
                        f'belong to shank 2',
            location=brain_regions[n],
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
        mat_loader = SingerLabMatLoader(subject_num, session_date)

        # extract recording times to epochs, same for both brain regions
        base_path = processed_data_folder / brain_regions[0]
        filenames = glob.glob(str(base_path) + '/0/eeg*.mat')  # use eeg files bc most consistent name across singer lab

        recording_epochs = get_recording_epochs(filenames, self.source_data['vr_files'], mat_loader)
        nwbfile.add_time_intervals(recording_epochs)

        # load probe and channel details and use to add electrodes
        probe_file_path = Path(self.source_data['channel_map_path'])
        probe_map = pd.read_csv(probe_file_path)
        num_electrodes = 0
        device = nwbfile.create_device(name=metadata['Ecephys']['Device'][0]['name'])  # TODO - search for MCU device
        for group in metadata['Ecephys']['ElectrodeGroup']:
            if group['name'] is not 'analog_inputs':
                nwbfile.create_electrode_group(name=group['name'],
                                               description=group['description'],
                                               location=group['location'],
                                               device=device)

                # generate electrodes and electrode region
                for index, row in probe_map.iterrows():
                    nwbfile.add_electrode(id=index + num_electrodes,
                                          x=np.nan, y=np.nan, z=np.nan,
                                          rel_x=row['X'], rel_y=row['Y'], rel_z=row['K'],
                                          reference='ground pellet',
                                          imp=np.nan,
                                          location=group['location'],
                                          filtering='none',
                                          group=nwbfile.electrode_groups[group['name']])

                num_electrodes = index + 1  # add 1 so that next loop creates new unique index

        # extract best ripple channel


        # extract filtered lfp signals
        signal_bands = ['eeg', 'EEG/beta', 'EEG/delta', 'EEG/theta', 'EEG/lowgamma', 'EEG/ripple']
        elec_regions = [0, 64, 128]
        for ind, br in enumerate(brain_regions):
            for band in signal_bands:
                # get channel directory list ordered by depth
                channel_dirs = []
                for ntrode, row in probe_map.iterrows():
                    channel = row['HW Channel']
                    channel_dirs.append(processed_data_folder / br / str(int(channel)))

                elec_table_region = nwbfile.create_electrode_table_region(region=list(range(elec_regions[ind], elec_regions[ind+1])),
                                                                          description=f'channels in {br}')

                # get metadata from first file (should be same for all)
                filenames = list(channel_dirs[0].glob(f'{band}?.mat'))  # TODO - get recs from spreadsheet
                recs = [f.stem.strip(band) for f in filenames]
                temp_data = mat_loader.run_conversion(filenames[0], recs[0], 'scipy')
                samp_rate = temp_data.samprate
                #filter = ''.join(temp_data.filteringinfo)
                #assert (temp_data.region == br)

                temp_data = mat_loader.run_conversion(filenames, recs, 'concat_array')
                data_array = np.array(temp_data)
                # TODO figure out the best way to extract the different data components of the filtered data

                # create data iterator
                num_steps = int(max(recording_epochs['stop_time'][:]) * samp_rate)
                data = MultiFileArrayIterator(channel_dirs, band, mat_loader, recs, num_steps)

                # general data interfaces as either as LFP or filtered ephys objects
                check_module(nwbfile, 'ecephys', 'contains processed extracellular electrophysiology data')
                if band == 'eeg':
                    band_name = f'lfp_{br}'
                    descript = 'local field potential data generated by singer lab preprocessing pipeline'
                    data_obj = LFP(name=band_name)
                else:
                    band = f'{band.strip("EEG/")}_{br}'
                    descript = 'filtered ephys data generated by singer lab preprocessing pipeline'
                    data_obj = FilteredEphys(name=band_name)

                # add electrical series to NWB file
                ecephys_ts = ElectricalSeries(name=band_name,
                                              data=H5DataIO(data, compression='gzip'),
                                              starting_time=0.0,
                                              #filtering=filter,
                                              rate=float(samp_rate),
                                              electrodes=elec_table_region,
                                              description=descript)
                data_obj.add_electrical_series(ecephys_ts)
                nwbfile.processing['ecephys'].add(data_obj)

        # extract lfp event times from best ripple channel
        signal_bands = ['thetas', 'nonthetas', 'ripples']
        for ind, br in enumerate(brain_regions):
            for band in signal_bands:
                filenames = list(channel_dirs[0].glob(f'{band}?.mat'))  # TODO - get recs from spreadsheet
                mat_loader = SingerLabMatLoader(filenames[0], subject_num, session_date, recs[0])
                temp_data = mat_loader.run_conversion('scipy')

                lfp_events = TimeIntervals(name='band',
                                             description='')

                lfp_events.add_column(name="behavior_task",
                                      description="indicates whether VR task was displayed on the screen (1), or if the"
                                                  " screen was left covered/blank as a baseline period (0)")
                # add events to nwb file
                nwbfile.processing['ecephys'].add(data_obj)

def get_recording_epochs(filenames, vr_files, mat_loader):
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
        rec = f.split('.mat')[0][-1]
        eeg_mat = mat_loader.run_conversion(f, rec, 'scipy')

        # add to nwb file
        duration = (len(eeg_mat.time) / eeg_mat.samprate)  # in seconds
        recording_epochs.add_row(start_time=start_time, stop_time=start_time + duration,
                                 behavior_task=vr_files[ind])
        start_time = start_time + duration

    return recording_epochs

