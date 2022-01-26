import numpy as np
import pandas as pd
import re

from datetime import timedelta, datetime
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
        # metadata_schema['properties']['SpatialSeries'] = get_schema_from_hdmf_class(SpatialSeries)
        # metadata_schema['required'].append('SpatialSeries')
        return metadata_schema

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # convert the mat file into a nested dict format
        mat_file = self.source_data['file_path']

        if Path(mat_file).is_file():
            # convert the mat file into dict and data frame format
            matin = convert_mat_file_to_dict(mat_file)
            virmen_df = pd.DataFrame(matin['virmenData']['data'], columns=matin['virmenData']['dataHeaders'])

            # create behavioral time series objects
            virmen_time = virmen_df["time"].apply(lambda x: matlab_time_to_datetime(x))
            session_start_time = virmen_time[0]
            timestamps = [(t - session_start_time).total_seconds() for t in virmen_time]

            pos_obj, trans_vel_obj, rot_vel_obj, view_angle_obj = create_behavioral_time_series(virmen_df, timestamps)

            # add behavioral time series to processing module
            check_module(nwbfile, 'behavior', 'contains processed behavioral data')
            nwbfile.processing['behavior'].add(pos_obj)
            nwbfile.processing['behavior'].add(trans_vel_obj)
            nwbfile.processing['behavior'].add(rot_vel_obj)
            nwbfile.processing['behavior'].add(view_angle_obj)

            # add behavioral events to processing module
            licks = np.diff(virmen_df['numLicks'], prepend=virmen_df['numLicks'][0])
            lick_obj = BehavioralEvents(name='licks')
            lick_ts = TimeSeries(name='licks',
                       data=H5DataIO(licks, compression="gzip"),
                       description='licking events detected by photointerrupter',
                       unit='au',
                       resolution=np.nan,
                       timestamps=H5DataIO(timestamps, compression="gzip")
                       )
            lick_obj.add_timeseries(lick_ts)
            nwbfile.processing['behavior'].add(lick_obj)

            rewards = np.diff(virmen_df['numRewards'], prepend=virmen_df['numRewards'][0])
            reward_obj = BehavioralEvents(name='rewards')
            reward_ts = TimeSeries(name='rewards',
                               data=H5DataIO(rewards, compression="gzip"),
                               description='reward delivery times',
                               unit='au',
                               resolution=np.nan,
                               timestamps=H5DataIO(timestamps, compression="gzip")
                               )
            reward_obj.add_timeseries(reward_ts)
            nwbfile.processing['behavior'].add(reward_obj)

            # add behavioral intervals with epoch and trial tables
            # get trial start and end times
            task_state_names = (
                'trial_start', 'initial_cue', 'update_cue', 'delay_cue', 'choice_made', 'reward', 'trial_end',
                'inter_trial')
            task_state_dict = dict(zip(task_state_names, range(1, 9)))
            trial_starts = virmen_df.index[virmen_df.taskState == task_state_dict['trial_start']].to_numpy()
            trial_ends = virmen_df.index[virmen_df.taskState == task_state_dict['trial_end']].to_numpy()

            # add all trials and corresponding info to nwb file
            for k in range(len(trial_starts)):
                nwbfile.add_trial(start_time=timestamps[trial_starts[k]], stop_time=timestamps[trial_ends[k]])

            maze_names = ('linear', 'ymaze_short', 'ymaze_long', 'ymaze_delay')
            maze_names_dict = dict(zip(maze_names, range(1, 5)))

            nwbfile.add_trial_column(name='id', description='number of trial in session', data=range(trial_starts))
            nwbfile.add_trial_column(name='maze_id', description='maze trial occurred in', data=test)
            nwbfile.add_trial_column(name='turn_type', description='left or right trial', data=test)
            nwbfile.add_trial_column(name='update_type', description='no-update, stay, or switch trial', data=temp)
            nwbfile.add_trial_column(name='choice', description='choice animal made', data=temp)
            nwbfile.add_trial_column(name='correct', description='whether trial was correct or not', data=temp)
            nwbfile.add_trial_column(name='duration', description='duration from start to end', data=temp)
            nwbfile.add_trial_column(name='iterations', description='number of samples/vr frames', data=temp)
            nwbfile.add_trial_column(name='i_delay', description='index when delay started', data=temp)
            nwbfile.add_trial_column(name='i_update', description='index when update started', data=temp)
            nwbfile.add_trial_column(name='i_delay2', description='index when second delay started', data=temp)
            nwbfile.add_trial_column(name='i_choice_made', description='index when choice made', data=temp)
            nwbfile.add_trial_column(name='delay_location', description='y-position on track where delay occurred',
                                     data=temp)
            nwbfile.add_trial_column(name='update_location', description='y-position on track where update occurred',
                                     data=temp)

            #
            # pattern_to_match = str(task_states_dict['interTrial']) + str(task_states_dict['startOfTrial'])
            # task_states_str = virmen_df['taskState'].astype(int).astype(str)
            # trial_starts_iloc = np.array([])
            # for match in re.finditer(pattern_to_match, task_states_long):
            #     trial_starts_iloc = np.append(trial_starts_iloc, (match.end()))
            #
            # for m in re.finditer(pattern_to_match, task_states_long, re.I):
            #     print(m.group(1))
            #     trial_starts_iloc = m.end()
            #
            # trial_starts_iloc
            # trial_starts = re.findall('2', task_states_long, re.I).findall(pattern_to_match)

            # add epochs for the different mazes and phases


def create_behavioral_time_series(df, timestamps):
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

    # make view angle objection
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

    return pos_obj, trans_velocity_obj, rot_velocity_obj, view_angle_obj
