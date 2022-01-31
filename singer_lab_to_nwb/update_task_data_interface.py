import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile, TimeSeries
from pynwb.behavior import SpatialSeries, Position, CompassDirection, BehavioralEvents
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class

from mat_conversion_utils import convert_mat_file_to_dict, create_indexed_array
from update_task_conversion_utils import matlab_time_to_datetime, check_module


class UpdateTaskVirmenInterface(BaseDataInterface):
    """
    Data interface for virmen data acquired with the update task

    See https://github.com/catalystneuro/tank-lab-to-nwb/blob/main/tank_lab_to_nwb/convert_towers_task/virmenbehaviordatainterface.py
    for a good example to get started with as I build this
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        return dict(
            required=['file_path'],
            properties=dict(
                file_path=dict(type='string')
            )
        )

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # convert the mat file into a nested dict format
        mat_file = self.source_data['file_path']

        if Path(mat_file).is_file():
            # convert the mat file into dict and data frame format
            matin = convert_mat_file_to_dict(mat_file)
            virmen_df = pd.DataFrame(matin['virmenData']['data'], columns=matin['virmenData']['dataHeaders'])

            # create behavioral time series objects and add to processing module
            virmen_time = virmen_df["time"].apply(lambda x: matlab_time_to_datetime(x))
            session_start_time = virmen_time[0]
            timestamps = [(t - session_start_time).total_seconds() for t in virmen_time]

            time, pos_obj, trans_vel_obj, rot_vel_obj, view_angle_obj = create_behavioral_time_series(virmen_df, timestamps)

            check_module(nwbfile, 'behavior', 'contains processed behavioral data')
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
            # remove the first trial and last trial bc incomplete and VR environment was likely obscured
            trial_starts = virmen_df.index[virmen_df.taskState == task_state_dict['trial_start']].to_numpy() + 1
            trial_starts = trial_starts[:-1]
            trial_ends = virmen_df.index[virmen_df.taskState == task_state_dict['trial_end']].to_numpy()
            trial_ends = trial_ends[1:]

            # add all trials and corresponding info to nwb file
            for t in range(len(trial_starts)):
                nwbfile.add_trial(start_time=timestamps[trial_starts[t]], stop_time=timestamps[trial_ends[t]])

            # trial maze ids
            maze_names = ('linear', 'ymaze_short', 'ymaze_long', 'ymaze_delay')
            maze_names_dict = dict(zip(maze_names, range(1, 5)))
            maze_ids = virmen_df['currentWorld'][trial_starts].to_numpy()
            nwbfile.add_trial_column(name='maze_id', description=f'{maze_names_dict}', data=maze_ids)

            # trial types l/r, none/stay/switch
            trial_types_dict = dict(zip(('right', 'left'), range(1,3)))  # flip right and left because projector flips
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
            durations_ind = trial_ends - trial_starts
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

def create_behavioral_time_series(df, timestamps):
    # make time object
    time = TimeSeries(name='time',
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
                             unit='au',
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
                               unit='radians',
                               reference_frame='left is positive, and right is negative because the projector flips '
                                               'the display on the x-axis',
                               resolution=np.nan,
                               timestamps=H5DataIO(timestamps, compression="gzip")
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