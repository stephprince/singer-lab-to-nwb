import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, ElectricalSeries, LFP
from pynwb.epoch import TimeIntervals
from pynwb.misc import DecompositionSeries
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class

from array_iterators import MultiFileArrayIterator, MultiDimMultiFileArrayIterator
from singer_lab_mat_loader import SingerLabMatLoader
from update_task_conversion_utils import get_module


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
        brain_regions = self.source_data['session_info'][['RegAB', 'RegCD']].values[0]
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
        )

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False, skip_decomposition: bool=False):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # get session info from the session info data frame
        session_info = self.source_data['session_info']
        subject_num = session_info['Animal'].values[0]
        session_date = session_info['Date'].values[0]
        brain_regions = session_info[['RegAB', 'RegCD']].values[0]
        vr_files = session_info[['Behavior']].values.squeeze()
        rec_files = session_info[['Recording']].values.squeeze().tolist()

        mat_loader = SingerLabMatLoader(subject_num, session_date)

        # extract recording times to epochs, same for both brain regions
        processed_data_folder = Path(self.source_data['processed_data_folder'])
        base_path = processed_data_folder / brain_regions[0]
        filenames = list(base_path.glob(f'0/eeg{rec_files}.mat'))  # use eeg files bc most consistent name across lab
        assert len(filenames) == len(rec_files), "Number of eeg files does not match expected number of recordings"

        recording_epochs = get_recording_epochs(filenames, vr_files, mat_loader)
        nwbfile.add_time_intervals(recording_epochs)

        # if stub_test, cut file to shortest recording duration
        if stub_test:
            durations = np.array(recording_epochs['stop_time']) - np.array(recording_epochs['start_time'])
            shortest_rec = rec_files[int(np.where(durations == min(durations))[0])]
            rec_files = [str(shortest_rec)]
            nwbfile.intervals['recording_epochs'].add_column(name='stub_test_includes',
                                                             description='indicates which recording was used to '
                                                                         'generate a stub test for testing purposes',
                                                             data=durations == min(durations))

        # load probe and channel details and use to add electrodes
        probe_file_path = Path(self.source_data['channel_map_path'])
        probe_map = pd.read_csv(probe_file_path)

        num_electrodes = 0
        for group in metadata['Ecephys']['ElectrodeGroup']:
            # add device and electrode group if needed
            device_descript = [d['description'] for d in metadata['Ecephys']['Device'] if d['name'] == group['device']]
            if group['device'] not in nwbfile.devices:
                device = nwbfile.create_device(name=group['device'], description=str(device_descript[0]))
            nwbfile.create_electrode_group(name=group['name'],
                                           description=group['description'],
                                           location=group['location'],
                                           device=device)

            # generate electrodes from probe
            if group['name'] is not 'analog_inputs':
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

        # get hardware channel info and list of channel directories sorted by depth to use for mapping
        channel_dirs = []
        hw_channels = []
        for br in brain_regions:  # get channel dirs ordered by depth first brain region and then second
            for ntrode, row in probe_map.iterrows():
                channel = row['HW Channel']
                hw_channels.append(int(channel))
                channel_dirs.append(processed_data_folder / br / str(int(channel)))
        assert(len(channel_dirs) == len(brain_regions)*len(probe_map))

        nwbfile.add_electrode_column(name='hardware_channel',
                                     description='channel ID from hardware wiring, used as folder names for singer '
                                                 'lab preprocessed mat data',
                                     data=hw_channels)

        # add ripple channel information
        ripple_chan_filename = processed_data_folder / br / f'bestripplechan{subject_num}_{session_date}'
        ripple_chan_data = get_ripple_channel(ripple_chan_filename, nwbfile, mat_loader)

        nwbfile.add_electrode_column(name='ripple_channel',
                                     description='channel with the largest mean power in the ripple band, used as '
                                                 'channel for detecting',
                                     data=ripple_chan_data)

        # extract lfp
        elec_table_region = nwbfile.create_electrode_table_region(region=list(range(len(channel_dirs))),
                                                                  description='neuronexus_probes')
        lfp_obj = get_lfp(channel_dirs, mat_loader, rec_files, elec_table_region)  # get lfp data object

        get_module(nwbfile, 'ecephys', 'contains processed extracellular electrophysiology data')
        nwbfile.processing['ecephys'].add(lfp_obj)

        # extract filtered lfp bands (beta, delta, theta, gamma, ripple)
        if not stub_test and not skip_decomposition:
            lfp_ts = nwbfile.processing['ecephys']['LFP']['LFP']
            amp_obj = get_lfp_decomposition(channel_dirs, mat_loader, rec_files, lfp_ts, 'amplitude', 'uV')
            phase_obj = get_lfp_decomposition(channel_dirs, mat_loader, rec_files, lfp_ts, 'phase', 'degrees')
            envelope_obj = get_lfp_decomposition(channel_dirs, mat_loader, rec_files, lfp_ts, 'envelope', 'uV')

            nwbfile.processing['ecephys'].add(amp_obj)
            nwbfile.processing['ecephys'].add(phase_obj)
            nwbfile.processing['ecephys'].add(envelope_obj)

        # extract lfp event times (thetas, nonthetas, ripples)
        elec_df = nwbfile.electrodes.to_dataframe()
        ripple_channel = elec_df['hardware_channel'][elec_df['ripple_channel'] == 1].values[0]
        rec_durations = nwbfile.intervals['recording_epochs'].to_dataframe()

        lfp_events_obj = get_lfp_events(processed_data_folder, 'CA1', ripple_channel, rec_durations, mat_loader, rec_files)
        for lfp_events in lfp_events_obj.values():
            nwbfile.processing['ecephys'].add(lfp_events)

        # extract digital and analog channels using the singer lab mat files
        analog_obj = get_analog_timeseries(nwbfile, processed_data_folder, mat_loader, rec_files)
        for analog_ts in analog_obj.values():
            nwbfile.add_acquisition(analog_ts)

        # extract digital signals (non-continuous, stored as time intervals)
        digital_obj = get_digital_events(processed_data_folder, rec_durations, mat_loader, rec_files)
        for digital_events in digital_obj.values():
            nwbfile.add_time_intervals(digital_events)


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
    recording_epochs.add_column(name="recording_number",
                                description="what the original recording file number this epoch was (as saved in the"
                                            "raw spikegadgets and .mat file names), 1-based indexing")

    # loop through files to get time intervals
    start_time = 0.0  # initialize start time to 0 sec
    for ind, f in enumerate(filenames):
        # load singer mat structure
        rec = str(f).split('.mat')[0][-1]
        eeg_mat = mat_loader.run_conversion(f, rec, 'scipy')

        # add to nwb file
        duration = (len(eeg_mat.time) / eeg_mat.samprate)  # in seconds
        recording_epochs.add_row(start_time=start_time, stop_time=start_time + duration,
                                 behavior_task=vr_files[ind], recording_number=rec)
        start_time = start_time + duration

    return recording_epochs


def get_lfp(channel_dirs, mat_loader, rec_files, elec_table_region):
    # get metadata from first file (should be same for all)
    filenames = list(channel_dirs[0].glob(f'eeg{rec_files}.mat'))
    assert len(filenames) == len(rec_files)
    recs = [f.stem.strip('eeg') for f in filenames]
    metadata = mat_loader.run_conversion(filenames[0], recs[0], 'scipy')
    temp_data = mat_loader.run_conversion(filenames, recs, 'concat_array')
    num_samples = len(temp_data)

    # general data interfaces as either as LFP or filtered ephys objects
    band_name = 'LFP'
    descript = f"local field potential data generated by singer lab preprocessing pipeline - " \
               f"{' '.join(''.join(metadata.descript).split())}"
    filter = ' '.join(''.join(metadata.filteringinfo).split())
    samp_rate = metadata.samprate

    # create data iterator
    data = MultiFileArrayIterator(channel_dirs, 'eeg', mat_loader, recs, num_samples)

    # add electrical series to NWB file
    data_obj = LFP(name=band_name)
    ecephys_ts = ElectricalSeries(name=band_name,
                                  data=H5DataIO(data, compression='gzip'),
                                  starting_time=0.0,
                                  # filtering=filter,
                                  rate=float(samp_rate),
                                  electrodes=elec_table_region,
                                  description=descript,
                                  conversion=0.001,  # data are in uV so multiply by 0.001 to get base units of volt
                                  comments=f'includes original 1-based recording file numbers: {rec_files}'
                                  )
    data_obj.add_electrical_series(ecephys_ts)

    return data_obj


def get_lfp_decomposition(channel_dirs, mat_loader, rec_files, lfp_ts, metric, units):
    freq_bands = ['delta', 'theta', 'beta', 'lowgamma', 'ripple']
    limits = [(1, 4), (4, 12), (12, 30), (20, 50), (150, 250)]  # TODO - don't hardcode and pull from somewhere?

    # get metadata from first channel and file and band (should be same for all)
    filenames = list(channel_dirs[0].glob(f'EEG/{freq_bands[0]}{rec_files}.mat'))
    metadata = mat_loader.run_conversion(filenames[0], rec_files[0], 'scipy')
    samp_rate = metadata.samprate
    temp_data = mat_loader.run_conversion(filenames, rec_files, 'concat_array')
    num_samples = len(temp_data)

    # get description of each filter
    descript = f"filtered ephys {metric} data generated by singer lab preprocessing pipeline"
    for band in freq_bands:
        filenames = list(channel_dirs[0].glob(f'EEG/{band}{rec_files}.mat'))
        assert len(filenames) == len(rec_files)
        metadata = mat_loader.run_conversion(filenames[0], rec_files[0], 'scipy')
        filter_descript = f"{band} {' '.join(''.join(metadata.descript).split())}"
        descript = "\n".join([descript, filter_descript])

    # find row matching metric of interest
    data_fields = [f[:-1] for f in metadata.fields.split()]
    metric_row = [ind for ind, m in enumerate(data_fields) if metric in m]

    # create data iterator
    dim_names = [f'EEG/{band}' for band in freq_bands]
    data = MultiDimMultiFileArrayIterator(channel_dirs, dim_names, mat_loader, rec_files, num_samples, metric_row)

    # add electrical series to NWB file
    decomp_series = DecompositionSeries(name=f'decomposition_{metric}',
                                        data=H5DataIO(data, compression='gzip'),
                                        metric=metric,
                                        starting_time=0.0,
                                        rate=float(samp_rate),
                                        source_timeseries=lfp_ts,
                                        description=descript,
                                        unit=units,
                                        comments=f'includes original 1-based recording file numbers: {rec_files}')

    # add info about frequency bands and limits
    for band, lim in zip(freq_bands, limits):
        decomp_series.add_band(band_name=band, band_limits=lim)

    return decomp_series


def get_ripple_channel(ripple_chan_filename, nwbfile, mat_loader):
    # load ripple channel data structure
    ripple_dict = mat_loader.run_conversion(ripple_chan_filename, [], 'dict')

    # extract ripple channel from electrode table
    elec_df = nwbfile.electrodes.to_dataframe()
    index = elec_df.index[(elec_df['hardware_channel'] == int(ripple_dict['channel'])) & (elec_df['location'] == 'CA1')]

    # create data structure for all of the channels
    ripple_channel_data = np.empty(np.shape(elec_df['hardware_channel']))
    ripple_channel_data.fill(np.nan)
    ripple_channel_data[index] = 1  # replace column with ripple channel as 1

    return ripple_channel_data


def get_lfp_events(processed_data_folder, br, channel, rec_durations, mat_loader, rec_files):
    signal_bands = ['thetas', 'nonthetas', 'ripples']
    lfp_events = dict.fromkeys(signal_bands)
    for band in signal_bands:
        # setup base interval structure
        events = TimeIntervals(name=f'{band}',
                               description=f'events identified from filtered lfp signals')

        column_dict = {'start_time': 'start of lfp event',
                       'stop_time': 'stop of lfp event',
                       'mid_time': 'middle of lfp event',
                       'start_ind': 'index of start time in lfp timeseries',
                       'stop_ind': 'index of stop time in lfp timeseries',
                       'mid_ind': 'index of middle time in lfp timeseries',
                       'energy': 'total sum squared energy of the waveform',
                       'peak': 'peak height/energy of the waveform',
                       'max_thresh': 'the largest threshold in stdev units at which this ripple would be detected',
                       'baseline': 'baseline value for event detection',
                       'threshold': 'baseline value with X number of std above the mean for event detection',
                       'std': 'standard deviation value used to generate threshold value',
                       'min_duration': 'time (in seconds) signal must be above threshold for event detection',
                       'excluded': 'any post-event detection exclusion criteria applied'}
        for key, value in column_dict.items():
            try:
                events.add_column(name=key, description=value)
            except ValueError:  # skip if it already exists
                pass

        # loop through files to add events
        filenames = list((processed_data_folder / br / str(channel)).glob(f'{band}{rec_files}.mat'))
        assert len(filenames) == len(rec_files)
        for ind, file in enumerate(filenames):
            # load up the data
            matin = mat_loader.run_conversion(file, file.stem.strip(band), 'scipy')
            num_events = np.size(matin.startind)
            samp_rate = matin.samprate or 2000  # TODO - make this not hardcoded, time*samples to adjust index

            # get values that occur for each event
            temp_data = []
            fields = ['starttime', 'endtime', 'midtime', 'startind', 'endind', 'midind', 'energy', 'peak', 'maxthresh',
                      'baseline', 'threshold', 'std', 'minimum_duration', 'excluderipples']
            for field in fields:
                field_data = getattr(matin, field, np.nan)  # set default to nan if missing

                if field in ['startind', 'endind', 'midind']:
                    field_data = field_data + (rec_durations['start_time'][ind] * samp_rate)
                elif (field in ['starttime', 'endtime', 'midtime']) and (
                        field_data is not np.nan):  # catch for structures that only have start ind and not time
                    field_data = field_data + rec_durations['start_time'][ind]
                elif (field in ['starttime', 'endtime', 'midtime']) and (field_data is np.nan):
                    base_field = field.replace('time', 'ind')
                    field_data = (getattr(matin, base_field) / samp_rate) + rec_durations['start_time'][ind]

                if np.size(field_data) < num_events:  # resize if there is only one value
                    field_data = [field_data] * num_events

                temp_data.append(field_data)

            # get values that are different across events
            event_data = np.array(temp_data).T

            # loop through data to add each as row
            event_data = event_data if np.ndim(event_data) > 1 else [event_data]
            event_df = pd.DataFrame(event_data, columns=column_dict.keys())
            event_dict = event_df.to_dict('records')
            for e in event_dict:
                events.add_row(e)

        # add to output data structure
        lfp_events[band] = events

    return lfp_events


def get_analog_timeseries(nwbfile, data_folder, mat_loader, rec_files):
    # define analog signals saved in data
    analog_descript_dict = {'licks': 'signal from photointerrupter circuit, values of 5V indicate the mouse tongue is '
                                     'has broken through the laser beam, values of 0V indicate baseline',
                            'rotVelocity': 'movement of the spherical treadmill along the roll axis as measured by an '
                                           'optical gaming mouse and filtered through a lab view function, maximum '
                                           'values of -10 and 10V',
                            'transVelocity': 'movement of the spherical treadmill along the pitch axis as measured by '
                                             'an optical gaming mouse and filtered through a lab view function, maximum'
                                             ' values of -10 and 10V'}
    analog_chan_dict = {'licks': 1, 'rotVelocity': 2, 'transVelocity': 4}  # TODO - have this as a project input

    # look through signals and generate electrical series
    analog_obj = dict.fromkeys(analog_descript_dict.keys())
    for name, chan in analog_chan_dict.items():
        # concatenate analog signal across recordings
        analog_filenames = list(data_folder.glob(f'{name}{rec_files}.mat'))
        assert len(analog_filenames) == len(rec_files)
        analog_data = mat_loader.run_conversion(analog_filenames, rec_files, 'concat_array')
        metadata = mat_loader.run_conversion(analog_filenames[0], rec_files[0], 'scipy')

        # generate electrodes and electrode region
        last_electrode = max(nwbfile.electrodes['id'])
        nwbfile.add_electrode(id=last_electrode + 1,
                              x=np.nan, y=np.nan, z=np.nan,
                              rel_x=np.nan, rel_y=np.nan, rel_z=np.nan,
                              imp=np.nan,
                              reference='none',
                              location='none',
                              filtering='none',
                              ripple_channel=np.nan, hardware_channel=metadata.adcChannel,
                              group=nwbfile.electrode_groups['analog_inputs'])
        elec_table_region = nwbfile.create_electrode_table_region(region=[last_electrode + 1],
                                                                  description=f'{metadata.descript} {name}')

        # general electrical series objects
        analog_obj[name] = ElectricalSeries(name=name,
                                            data=H5DataIO(analog_data, compression='gzip'),
                                            starting_time=0.0,
                                            rate=float(metadata.samprate),
                                            electrodes=elec_table_region,
                                            description=analog_descript_dict[name],
                                            comments=f'includes original 1-based recording file numbers: {rec_files}')
    return analog_obj


def get_digital_events(data_folder, rec_durations, mat_loader, rec_files):
    # define digital signals and channel dicts
    dig_descript_dict = {'sync': 'synchronizing pulse sent from virmen software, from spikegadgets digital signals',
                         'trial': 'indicates when trial is occurring vs inter trial, from spikegadgets digital signals',
                         'update': 'indicates when the update cue is on or not, from spikegadgets digital signals',
                         'delay': 'indicates whether the delay cue is on or not, from spikegadgets digital signals'}
    dig_chan_dict = {'sync': 1, 'trial': 2, 'update': 3, 'delay': 4}  # TODO - have this as a project input

    # loop through signals and generate electrical series
    digital_obj = dict.fromkeys(dig_descript_dict.keys())
    for name, chan in dig_chan_dict.items():

        # get event times from all files adjusted for time
        digital_filenames = list(data_folder.glob(f'{name}{rec_files}.mat'))
        assert len(digital_filenames) == len(rec_files)
        all_on_times = []
        all_off_times = []
        for ind, file in enumerate(digital_filenames):
            # load up the data
            matin = mat_loader.run_conversion(file, file.stem.strip(name), 'scipy')
            samp_rate = float(matin.samprate)

            # get periods where digital channel is ON
            start_time = float(matin.start_time)
            if not isinstance(matin.state, int):  # check if there's only one value then no signals acquired
                on_samples = [matin.data[ind] for ind, state in enumerate(matin.state) if state == 1]
                off_samples = [matin.data[ind] for ind, state in enumerate(matin.state) if state == 0]

                # clean up times for start and end of file
                if matin.state[0] == 0:  # if first value is off, remove first off sample
                    off_samples = off_samples[1:]
                elif matin.state[0] == 1:  # if first value is on, remove first on+off sample bc don't know actual start
                    on_samples = on_samples[1:]
                    off_samples = off_samples[1:]

                if matin.state[-1] == 1:  # if last value is on, remove last on sample bc we don't know actual end
                    on_samples = on_samples[:-1]

                # convert to seconds from start time, adjusted for duration
                on_times = (np.array(on_samples) / samp_rate) - start_time  # in seconds
                off_times = (np.array(off_samples) / samp_rate) - start_time  # in seconds

                # append to list and adjust for duration
                all_on_times.extend(on_times + rec_durations['start_time'][ind])  # adjust for durations up to that point
                all_off_times.extend(off_times + rec_durations['start_time'][ind])  # adjust for durations up to that point

        # make time intervals structure for each signal
        digital_obj[name] = TimeIntervals(name=name, description=dig_descript_dict[name])

        assert (len(all_on_times) == len(all_off_times))
        for start, stop in zip(all_on_times, all_off_times):
            digital_obj[name].add_row(start_time=start, stop_time=stop)

    return digital_obj
