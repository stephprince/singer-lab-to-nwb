import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection, BehavioralEvents
from pynwb.epoch import TimeIntervals
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema

from update_task_conversion_utils import get_module
from mat_conversion_utils import convert_mat_file_to_dict, matlab_time_to_datetime
from singer_lab_mat_loader import SingerLabMatLoader


class UpdateTaskVirmenInterface(BaseDataInterface):
    """
    Data interface for virmen data acquired with the update task
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        return dict(
            required=['file_path'],
            properties=dict(
                file_path=dict(type='string'),
                session_id=dict(type='string'),
                synced_file_path=dict(type='string'),
            )
        )

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # load virmen files for the whole session
        virmen_df_list = self.load_virmen_data()
        virmen_df = pd.concat(virmen_df_list, axis=0, ignore_index=True)

        # get timestamps for all of the virmen data
        subject_num = metadata['Subject']['subject_id'][1:]
        session_date = self.source_data['session_id'].split(f'{subject_num}_')[1]
        mat_loader = SingerLabMatLoader(subject_num, session_date)
        timestamps_list = self.align_virmen_timestamps(mat_loader, virmen_df_list)
        timestamps = [t for sublist in timestamps_list for t in sublist]

        # get virmen intervals and add to processing module
        behavior_epochs = TimeIntervals(
            name="behavior_epochs",
            description='Indicates the start and stop times in which behavior data was both collected AND '
                        'synchronized to ephys data.')
        for ts in timestamps_list:
            behavior_epochs.add_row(start_time=ts.values[0], stop_time=ts.values[-1])

        get_module(nwbfile, 'behavior', 'contains processed behavioral data')
        nwbfile.add_time_intervals(behavior_epochs)

        # create behavioral time series objects and add to processing module
        time, pos_obj, trans_vel_obj, rot_vel_obj, view_angle_obj = create_behavioral_time_series(virmen_df, timestamps)

        nwbfile.processing['behavior'].add(time)
        nwbfile.processing['behavior'].add(pos_obj)
        nwbfile.processing['behavior'].add(trans_vel_obj)
        nwbfile.processing['behavior'].add(rot_vel_obj)
        nwbfile.processing['behavior'].add(view_angle_obj)

        # create behavioral events objects and add to processing module
        lick_obj, reward_obj = create_behavioral_events(virmen_df, timestamps)

        nwbfile.processing['behavior'].add(lick_obj)
        nwbfile.processing['behavior'].add(reward_obj)

        # add behavioral trial info
        task_state_names = (
            'trial_start', 'initial_cue', 'update_cue', 'delay_cue', 'choice_made', 'reward', 'trial_end',
            'inter_trial')
        task_state_dict = dict(zip(task_state_names, range(1, 9)))

        # add 1 to trial start times bc that is when teleportation occurs and world switches to new one and
        t_starts = virmen_df.index[virmen_df.taskState == task_state_dict['trial_start']].to_numpy() + 1
        t_ends = virmen_df.index[virmen_df.taskState == task_state_dict['trial_end']].to_numpy()

        # clean up times for start and end of file and file transition points
        file_switch_inds = [len(df) for df in virmen_df_list]
        start_ind = 0
        trial_starts = []
        trial_ends = []
        for switch_ind in file_switch_inds:
            t_starts_temp = t_starts[(t_starts > start_ind) & (t_starts < start_ind + switch_ind)]
            t_ends_temp = t_ends[(t_ends > start_ind) & (t_ends < start_ind + switch_ind)]

            # if first trial end time is before the first start, remove first end
            if t_starts_temp[0] > t_ends_temp[0]:
                t_ends_temp = t_ends_temp[1:]

            # if last trial start is after trial_end, remove last trial start
            if t_ends_temp[-1] < t_starts_temp[-1]:
                t_starts_temp = t_starts_temp[:-1]

            start_ind = start_ind + switch_ind  # add file iteration lengths
            trial_starts.extend(t_starts_temp)
            trial_ends.extend(t_ends_temp)

        # add all trials and corresponding info to nwb file
        for start, end in zip(trial_starts, trial_ends):
            nwbfile.add_trial(start_time=timestamps[start], stop_time=timestamps[end])

        # trial maze ids
        maze_names = ('linear', 'ymaze_short', 'ymaze_long', 'ymaze_delay')
        maze_names_dict = dict(zip(maze_names, range(1, 5)))
        maze_ids = virmen_df['currentWorld'][trial_starts].to_numpy()
        nwbfile.add_trial_column(name='maze_id', description=f'{maze_names_dict}', data=maze_ids)

        # trial types l/r, none/stay/switch
        trial_types_dict = dict(zip(('right', 'left'), range(1, 3)))  # flip right and left because projector flips
        update_names_dict = dict(zip(('nan', 'switch', 'stay'), range(1, 4)))
        trial_types = virmen_df['trialType'][trial_starts].to_numpy()
        update_types = virmen_df['trialTypeUpdate'][trial_starts].to_numpy()
        nwbfile.add_trial_column(name='turn_type', description=f'{trial_types_dict}', data=trial_types)
        nwbfile.add_trial_column(name='update_type', description=f'{update_names_dict}', data=update_types)

        # trial choices
        choice_types = virmen_df['currentZone'][trial_ends].to_numpy()
        correct_flags = virmen_df['choice'][trial_ends].to_numpy()
        nwbfile.add_trial_column(name='choice', description='choice animal made', data=choice_types)
        nwbfile.add_trial_column(name='correct', description='whether trial was correct or not', data=correct_flags)

        # trial durations
        durations_sec = np.array(timestamps)[trial_ends] - np.array(timestamps)[trial_starts]
        durations_ind = np.array(trial_ends) - np.array(trial_starts) + 1  # add 1 for inclusive # of iterations
        nwbfile.add_trial_column(name='duration', description='duration in seconds', data=durations_sec)
        nwbfile.add_trial_column(name='iterations', description='number of samples/vr frames', data=durations_ind)

        # indices, times, and locations of delay, update, and choice events
        # use the updateOccurred columns bc these are closer to when the cues actually switch on/off in the VR
        delays = np.where(np.diff(virmen_df['delayUpdateOccurred'], prepend=0) == 1)[0]
        updates = np.where(np.diff(virmen_df['updateOccurred'], prepend=0) == 1)[0]
        choices = virmen_df.index[virmen_df.taskState == task_state_dict['choice_made']].to_numpy()

        i_delays, t_delays, loc_delays = get_task_event_times(virmen_df, delays, trial_starts, trial_ends,
                                                              timestamps)
        i_updates, t_updates, loc_updates = get_task_event_times(virmen_df, updates, trial_starts, trial_ends,
                                                                 timestamps)
        i_delays2, t_delays2, loc_delays2 = get_task_event_times(virmen_df, delays, trial_starts, trial_ends,
                                                                 timestamps, event_ind=1)
        i_choices, t_choices, loc_choices = get_task_event_times(virmen_df, choices, trial_starts, trial_ends,
                                                                 timestamps)

        nwbfile.add_trial_column(name='i_delay', description='index when delay started', data=i_delays)
        nwbfile.add_trial_column(name='i_update', description='index when update started', data=i_updates)
        nwbfile.add_trial_column(name='i_delay2', description='index when second delay started', data=i_delays2)
        nwbfile.add_trial_column(name='i_choice_made', description='index when choice made', data=i_choices)

        nwbfile.add_trial_column(name='t_delay', description='time when delay started', data=t_delays)
        nwbfile.add_trial_column(name='t_update', description='time when update started', data=t_updates)
        nwbfile.add_trial_column(name='t_delay2', description='time when second delay started', data=t_delays2)
        nwbfile.add_trial_column(name='t_choice_made', description='time when choice made', data=t_choices)

        nwbfile.add_trial_column(name='delay_location', description='y-position on track where delay occurred',
                                 data=loc_delays)
        nwbfile.add_trial_column(name='update_location', description='y-position on track where update occurred',
                                 data=loc_updates)
        nwbfile.add_trial_column(name='delay2_location', description='y-position on track where delay 2 occurred',
                                 data=loc_delays2)

    def load_virmen_data(self):
        virmen_df_list = []
        if self.source_data['synced_file_path']:  # if there exists ephys data, use that
            base_path = Path(self.source_data['synced_file_path'])
            recs = self.source_data['session_info']['Recording'].to_list()
            behavior_recs = self.source_data['session_info']['Behavior'].to_list()

            virmen_files = list(base_path.glob(f'virmenDataSynced{recs}.csv'))
            assert np.sum(behavior_recs) == len(virmen_files), 'Number of virmen files does not match expected number of ' \
                                                            'behavior sessions '

            # load up csv files
            for files in virmen_files:
                virmen_df = pd.read_csv(files)
                virmen_df.dropna(subset=['spikeGadgetsTimes'], inplace=True)  # remove data w/o corresponding ephys data
                virmen_df_list.append(virmen_df)

        else:  # otherwise use behavioral data
            base_path = Path(self.source_data['file_path'])
            virmen_files = list(base_path.glob(f'{session_id}*/virmenDataRaw.mat'))

            behavior_recs = self.source_data['' \
                                             ''][['Behavior']].values.squeeze().tolist()
            assert sum(behavior_recs) == len(virmen_files), 'Number of virmen files does not match expected number of ' \
                                                            'behavior sessions '

            # load up mat files
            for files in virmen_files:
                matin = convert_mat_file_to_dict(files)
                virmen_df = pd.DataFrame(matin['virmenData']['data'], columns=matin['virmenData']['dataHeaders'])
                virmen_df_list.append(virmen_df)

        return virmen_df_list

    def align_virmen_timestamps(self, mat_loader, virmen_df_list):
        if self.source_data['synced_file_path']:  # if there exists ephys data, use that
            # get base file paths
            base_path = Path(self.source_data['synced_file_path'])
            recs = self.source_data['session_info'][['Recording']].values.squeeze().tolist()
            all_eeg_files = list(base_path.glob(f'CA1/0/eeg{recs}.mat'))
            eeg_recs = [int(path.stem[-1]) for path in all_eeg_files]
            virmen_files = list(base_path.glob(f'virmenDataSynced{recs}.csv'))
            virmen_recs = [int(path.stem[-1]) for path in virmen_files]
            assert len(virmen_df_list) == len(virmen_recs), 'Number of virmen dataframes does not match number of recs'

            # get durations and start times of ALL recordings (including non-VR ones)
            start = 0.0  # initialize start time to 0 sec
            sg_start_times = np.empty((np.max(eeg_recs)))  # will be accessed with 0-based rec number so fill with nan
            sg_start_samples = np.empty((np.max(eeg_recs)))
            sg_start_times.fill(np.nan)  # spikegadgets start times
            sg_start_samples.fill(np.nan)  # spikegadgets start samples
            for file in all_eeg_files:
                eeg_mat = mat_loader.run_conversion(file, file.stem[-1], 'scipy')
                sg_start_times[int(file.stem[-1])-1] = start
                sg_start_samples[int(file.stem[-1])-1] = eeg_mat.time[0]
                start = start + (len(eeg_mat.time) / eeg_mat.samprate)  # get duration of each recording

            # calculate timestamps of virmen data based on these samples and durations
            sg_samp_rate = eeg_mat.samprate * eeg_mat.downsample
            timestamps = []
            virmen_recs_0_based = [int(r) - 1 for r in
                                   virmen_recs]  # adjust by 1 because 0-based indexing vs. file names
            for df, rec in zip(virmen_df_list, virmen_recs_0_based):
                sg_samples = df["spikeGadgetsTimes"]
                sg_time_elapsed = (sg_samples - sg_start_samples[
                    rec]) / sg_samp_rate  # since 1st ephys sample of that rec
                sg_time_elapsed = sg_time_elapsed + sg_start_times[rec]  # adjust for any previous recordings

                timestamps.append(sg_time_elapsed)

        else:  # otherwise use behavioral data
            session_start_time = virmen_df_list[0]["time"].apply(lambda x: matlab_time_to_datetime(x))[0]
            timestamps = []
            for df in virmen_df_list:
                virmen_time = df["time"].apply(lambda x: matlab_time_to_datetime(x))
                time_elapsed = [(t - session_start_time).total_seconds() for t in virmen_time]
                timestamps.append(time_elapsed)

        return timestamps


def create_behavioral_time_series(df, timestamps):
    # make time object
    time = TimeSeries(name='time',
                      description="time recorded by virmen software since session state time",
                      data=H5DataIO(timestamps, compression="gzip"),
                      unit='s',
                      resolution=np.nan,
                      timestamps=H5DataIO(timestamps, compression="gzip")
                      )
    # make position object
    pos_obj = Position(name="position")
    position = SpatialSeries(name='position',
                             data=df[['xPos', 'yPos']].values,
                             description='the x and y position values for the mouse in the virtual-reality '
                                         'environment. X is the first column and Y is the second',
                             reference_frame='forwards is positive, backwards is negative, left is positive, and '
                                             'right is negative because the projector flips the display on the '
                                             'x-axis',
                             resolution=np.nan,
                             timestamps=H5DataIO(timestamps, compression="gzip")
                             )
    pos_obj.add_spatial_series(position)

    # make velocity objects
    trans_velocity_obj = TimeSeries(name='translational_velocity',
                                    data=df['transVeloc'].values,
                                    description='forward and backwards velocity in the virtual reality '
                                                'environment. The pitch axis as tracked by the optical '
                                                'mouse.',
                                    unit='au',
                                    resolution=np.nan,
                                    timestamps=H5DataIO(timestamps, compression="gzip")
                                    )
    rot_velocity_obj = TimeSeries(name='rotational_velocity',
                                  data=H5DataIO(df['rotVeloc'].values, compression="gzip"),
                                  description='sideways velocity in the virtual reality environment. The '
                                              'roll axis as tracked by the optical mouse.',
                                  unit='au',
                                  resolution=np.nan,
                                  timestamps=H5DataIO(timestamps, compression="gzip")
                                  )

    # make view angle object
    view_angle_obj = CompassDirection(name="view_angle")
    view_angle = SpatialSeries(name="view_angle",
                               data=H5DataIO(df['viewAngle'].values, compression="gzip"),
                               description='direction of mouse in the virtual reality environment',
                               reference_frame='left is positive, and right is negative because the projector flips '
                                               'the display on the x-axis',
                               resolution=np.nan,
                               timestamps=H5DataIO(timestamps, compression="gzip",),
                               unit='degrees'
                               )
    view_angle_obj.add_spatial_series(view_angle)

    return time, pos_obj, trans_velocity_obj, rot_velocity_obj, view_angle_obj


def create_behavioral_events(df, timestamps):
    # make lick object
    licks = np.diff(df['numLicks'], prepend=df['numLicks'][0])
    lick_obj = BehavioralEvents(name='licks')
    lick_ts = TimeSeries(name='licks',
                         data=H5DataIO(licks, compression="gzip"),
                         description='licking events detected by photointerrupter',
                         unit='au',
                         resolution=np.nan,
                         timestamps=H5DataIO(timestamps, compression="gzip")
                         )
    lick_obj.add_timeseries(lick_ts)

    # make reward object
    rewards = np.diff(df['numRewards'], prepend=df['numRewards'][0])
    reward_obj = BehavioralEvents(name='rewards')
    reward_ts = TimeSeries(name='rewards',
                           data=H5DataIO(rewards, compression="gzip"),
                           description='reward delivery times',
                           unit='au',
                           resolution=np.nan,
                           timestamps=H5DataIO(timestamps, compression="gzip")
                           )
    reward_obj.add_timeseries(reward_ts)

    return lick_obj, reward_obj


def get_task_event_times(df, events, trial_starts, trial_ends, timestamps, event_ind=0):
    # indices of delay, update, and choice events
    i_events = []
    for t in range(len(trial_starts)):
        temp = events[np.logical_and(events > trial_starts[t], events < trial_ends[t])]
        i_events.append(temp[event_ind] if event_ind < len(temp) else np.nan)

    # times of delay, update, and choice events
    t_events = [timestamps[i] if i is not np.nan else i for i in i_events]

    # locations of delay and update events
    loc_events = [df['yPos'][i] if i is not np.nan else i for i in i_events]

    return i_events, t_events, loc_events
