from nwb_conversion_tools.basedatainterface import BaseDataInterface

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
        metadata_schema['properties']['SpatialSeries'] = get_schema_from_hdmf_class(SpatialSeries)
        metadata_schema['required'].append('SpatialSeries')
        return metadata_schema


    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Tank lab behavioral interface."""
        mat_file = self.source_data['file_path']
        matin = convert_mat_file_to_dict(mat_file)
        session_start_time = array_to_dt(matin['log']['session']['start'])

        # Intervals
        if Path(mat_file).is_file():
            nwbfile.add_epoch_column('label', 'name of epoch')

            if isinstance(matin['log']['block'], dict):
                epochs = [matin['log']['block']]
            else:
                epochs = matin['log']['block']

            trials = [trial for epoch in epochs for trial in epoch['trial'] if
                      not np.isnan(trial['start'])]

            # Extension of lab metadata
            experiment_metadata = matin['log']['version']
            subject_metadata = matin['log']['animal']

            rig_atrr = ['rig', 'simulationMode', 'hasDAQ', 'hasSyncComm', 'minIterationDT',
                        'arduinoPort', 'sensorDotsPerRev', 'ballCircumference', 'toroidXFormP1',
                        'toroidXFormP2', 'colorAdjustment', 'soundAdjustment', 'nidaqDevice',
                        'nidaqPort', 'nidaqLines', 'syncClockChannel', 'syncDataChannel',
                        'rewardChannel', 'rewardSize', 'rewardDuration', 'laserChannel',
                        'rightPuffChannel', 'leftPuffChannel', 'webcam_name']
            rig = dict((k, v) for k, v in experiment_metadata['rig'].items() if k in rig_atrr)
            rig.update((k, v.astype(np.int32)) for k, v in rig.items() if
                       isinstance(v, np.ndarray) and v.dtype == np.uint8)
            rig_extension = RigExtension(name='rig', **rig)

            maze_extension = MazeExtension(name='mazes',
                                           description='description of the mazes')

            for maze in experiment_metadata['mazes']:
                flatten_maze_dict = flatten_nested_dict(maze)
                maze_extension.add_row(**dict((k, v) for k, v in flatten_maze_dict.items()
                                              if k in MazeExtension.mazes_attr))

            num_trials = len(trials)
            session_end_time = array_to_dt(matin['log']['session']['end']).isoformat()
            converted_metadata = convert_function_handle_to_str(mat_file_path=mat_file)
            lab_meta_data = dict(
                name='LabMetaData',
                experiment_name=converted_metadata[
                    'experiment_name'] if 'experiment_name' in converted_metadata else '',
                world_file_name=experiment_metadata['name'],
                protocol_name=converted_metadata[
                    'protocol_name'] if 'protocol_name' in converted_metadata else '',
                stimulus_bank_path=subject_metadata['stimulusBank'] if subject_metadata[
                    'stimulusBank'] else '',
                commit_id=experiment_metadata['repository'],
                location=experiment_metadata['rig']['rig'],
                num_trials=num_trials,
                session_end_time=session_end_time,
                rig=rig_extension,
                mazes=maze_extension,
                # session_performance=0.,  # comment out to add (not in behavior file)
            )
            metadata['lab_meta_data'] = LabMetaDataExtension(**lab_meta_data)
            nwbfile.add_lab_meta_data(metadata['lab_meta_data'])

            epoch_start_dts = [array_to_dt(epoch['start']) for epoch in epochs]
            epoch_durations_dts = [timedelta(seconds=epoch['duration']) for epoch in epochs]
            epoch_start_nwb = [(epoch_start_dt - session_start_time).total_seconds()
                               for epoch_start_dt in epoch_start_dts]
            epoch_end_nwb = [(epoch_start_dt - session_start_time + epoch_duration).total_seconds()
                             for epoch_start_dt, epoch_duration in
                             zip(epoch_start_dts, epoch_durations_dts)]
            for j, (start, end) in enumerate(zip(epoch_start_nwb, epoch_end_nwb)):
                nwbfile.add_epoch(start_time=start, stop_time=end, label='Epoch' + str(j + 1))

            epoch_maze_ids = [epoch['mazeID'] for epoch in epochs]
            epoch_main_maze_ids = [epoch['mainMazeID'] for epoch in epochs]
            epoch_easy_flag = [epoch['easyBlockFlag'] for epoch in epochs]
            epoch_first_trial = [epoch['firstTrial'] for epoch in epochs]
            epoch_num_trials = []
            for epoch in epochs:
                num_non_empty_trials = 0
                for trial in epoch['trial']:
                    if not np.isnan(trial['start']):
                        num_non_empty_trials += 1
                epoch_num_trials.append(num_non_empty_trials)

            epoch_durations = [epoch['duration'] for epoch in epochs]
            epoch_reward_mil = [epoch['rewardMiL'] for epoch in epochs]
            epoch_stimulus_config = [epoch['stimulusConfig'] for epoch in epochs]
            nwbfile.add_epoch_column(name='maze_id',
                                     description='number of maze run in an epoch',
                                     data=epoch_maze_ids)
            nwbfile.add_epoch_column(name='main_maze_id',
                                     description='number of maze of "highest" level for subject',
                                     data=epoch_main_maze_ids)
            nwbfile.add_epoch_column(name='easy_epoch',
                                     description='1 if block was flagged as easy '
                                                 '(maze_id < main_maze_id)',
                                     data=epoch_easy_flag)
            nwbfile.add_epoch_column(name='first_trial',
                                     description='first trial run in an epoch',
                                     data=epoch_first_trial)
            nwbfile.add_epoch_column(name='num_trials',
                                     description='number of trials in an epoch',
                                     data=epoch_num_trials)
            nwbfile.add_epoch_column(name='duration',
                                     description='duration of epoch in seconds',
                                     data=epoch_durations)
            nwbfile.add_epoch_column(name='reward_ml',
                                     description='reward in ml',
                                     data=epoch_reward_mil)
            nwbfile.add_epoch_column(name='stimulus_config',
                                     description='stimulus configuration number',
                                     data=epoch_stimulus_config)

            trial_starts = [trial['start'] + epoch_start_nwb[0] for trial in trials]
            trial_durations = [trial['duration'] for trial in trials]
            trial_ends = [start_time + duration for start_time, duration in
                          zip(trial_starts, trial_durations)]
            for k in range(len(trial_starts)):
                nwbfile.add_trial(start_time=trial_starts[k], stop_time=trial_ends[k])

            nwbfile.add_trial_column(name='duration',
                                     description='duration of trial in seconds',
                                     data=trial_durations)
            trial_idx = [trial_id for num_trials in epoch_num_trials
                         for trial_id in np.arange(0, num_trials)]
            nwbfile.add_trial_column(name='trial_id',
                                     description='number of trial in block',
                                     data=trial_idx)
            trial_columns = [('iterations', 'number of iterations (frames) for entire trial'),
                             ('iCueEntry', 'iteration number when subject entered cue region'),
                             ('iMemEntry', 'iteration number when subject entered memory region'),
                             ('iTurnEntry', 'iteration number when subject entered turn region'),
                             ('iArmEntry', 'iteration number when subject entered arm region'),
                             ('iBlank', 'iteration number when screen is turned off'),
                             ('excessTravel', 'total distance traveled during the trial '
                                              'normalized to the length of the maze'),
                             ('rewardScale', 'multiplier of reward for each correct trial')]
            for column_name, desc in trial_columns:
                data = [trial[column_name] for trial in trials if column_name in trial]
                if data:
                    nwbfile.add_trial_column(name=column_name,
                                             description=desc,
                                             data=data)

            if 'trial_choice' in converted_metadata:
                nwbfile.add_trial_column(name='choice',
                                         description='choice (L=Left,R=Right,nil=Trial violation)',
                                         data=converted_metadata['trial_choice'])
            if 'trial_type' in converted_metadata:
                nwbfile.add_trial_column(name='trial_type',
                                         description='type of trial (L=Left,R=Right)',
                                         data=converted_metadata['trial_type'])

            # Processed cue timing and position
            left_cue_presence = [trial['cueCombo'][0] if len(trial['cueCombo'])
                                 else trial['cueCombo'] for trial in trials]
            left_cue_presence_data, left_cue_presence_indices = create_indexed_array(
                left_cue_presence)

            right_cue_presence = [trial['cueCombo'][1] if len(trial['cueCombo'])
                                  else trial['cueCombo'] for trial in trials]
            right_cue_presence_data, right_cue_presence_indices = create_indexed_array(
                right_cue_presence)

            left_cue_onsets = [
                trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOnset'][0] - 1]
                if np.any(trial['cueOnset'][0]) else trial['cueOnset'][0] for trial in trials]
            left_cue_onset_data, left_cue_onset_indices = create_indexed_array(left_cue_onsets)

            right_cue_onsets = [
                trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOnset'][1] - 1]
                if np.any(trial['cueOnset'][1]) else trial['cueOnset'][1] for trial in trials]
            right_cue_onset_data, right_cue_onset_indices = create_indexed_array(right_cue_onsets)

            left_cue_offsets = [
                trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOffset'][0] - 1]
                if np.any(trial['cueOffset'][0]) else trial['cueOffset'][0] for trial in trials]
            left_cue_offset_data, left_cue_offset_indices = create_indexed_array(left_cue_offsets)

            right_cue_offsets = [
                trial['start'] + epoch_start_nwb[0] + trial['time'][trial['cueOffset'][1] - 1]
                if np.any(trial['cueOffset'][1]) else trial['cueOffset'][1] for trial in trials]
            right_cue_offset_data, right_cue_offset_indices = create_indexed_array(right_cue_offsets)

            left_cue_positions = [trial['cuePos'][0] if len(trial['cuePos'])
                                  else trial['cuePos'] for trial in trials]
            left_cue_position_data, left_cue_position_indices = create_indexed_array(
                left_cue_positions)

            right_cue_positions = [trial['cuePos'][1] if len(trial['cuePos'])
                                   else trial['cuePos'] for trial in trials]
            right_cue_position_data, right_cue_position_indices = create_indexed_array(
                right_cue_positions)

            nwbfile.add_trial_column(name='left_cue_presence',
                                     description='indicates whether the nth cue appeared on the left',
                                     index=left_cue_presence_indices,
                                     data=left_cue_presence_data)

            nwbfile.add_trial_column(name='right_cue_presence',
                                     description='indicates whether the nth cue appeared on the right',
                                     index=right_cue_presence_indices,
                                     data=right_cue_presence_data)

            nwbfile.add_trial_column(name='left_cue_onset',
                                     description='onset times of left cues',
                                     index=left_cue_onset_indices,
                                     data=left_cue_onset_data)

            nwbfile.add_trial_column(name='right_cue_onset',
                                     description='onset times of right cues',
                                     index=right_cue_onset_indices,
                                     data=right_cue_onset_data)

            nwbfile.add_trial_column(name='left_cue_offset',
                                     description='offset times of left cues',
                                     index=left_cue_offset_indices,
                                     data=left_cue_offset_data)

            nwbfile.add_trial_column(name='right_cue_offset',
                                     description='offset times of right cues',
                                     index=right_cue_offset_indices,
                                     data=right_cue_offset_data)

            nwbfile.add_trial_column(name='left_cue_position',
                                     description='position of left cues',
                                     index=left_cue_position_indices,
                                     data=left_cue_position_data)

            nwbfile.add_trial_column(name='right_cue_position',
                                     description='position of right cues',
                                     index=right_cue_position_indices,
                                     data=right_cue_position_data)

            # Processed position, velocity, viewAngle
            pos_obj = Position(name="Position")
            view_angle_obj = CompassDirection(name='ViewAngle')

            timestamps = []
            pos_data = np.empty((0, 2))
            velocity_data = np.empty_like(pos_data)
            view_angle_data = []
            collision = []
            for trial in trials:
                trial_total_time = trial['start'] + epoch_start_nwb[0] + trial['time']
                timestamps.extend(trial_total_time.astype(np.float64, casting='same_kind'))

                padding = np.full((trial['time'].shape[0] - trial['position'].shape[0], 2), np.nan)
                trial_position = trial['position'][:, :-1]
                trial_velocity = trial['velocity'][:, :-1]
                trial_view_angle = trial['position'][:, -1]
                trial_collision = trial['collision']
                pos_data = np.concatenate([pos_data, trial_position, padding], axis=0)
                velocity_data = np.concatenate([velocity_data, trial_velocity, padding], axis=0)
                view_angle_data = np.concatenate([view_angle_data, trial_view_angle,
                                                  padding[:, 0]], axis=0)
                collision = np.concatenate([collision, trial_collision, padding[:, 0]], axis=0)
            time = TimeSeries(name='Time',
                              data=H5DataIO(timestamps, compression="gzip"),
                              unit='s',
                              resolution=np.nan,
                              timestamps=H5DataIO(timestamps, compression="gzip"))
            pos_obj.add_spatial_series(
                SpatialSeries(
                    name="SpatialSeries",
                    data=H5DataIO(pos_data, compression="gzip"),
                    reference_frame="unknown",
                    conversion=0.01,
                    resolution=np.nan,
                    timestamps=H5DataIO(timestamps, compression="gzip")
                )
            )
            velocity_ts = TimeSeries(name='Velocity',
                                     data=H5DataIO(velocity_data, compression="gzip"),
                                     unit='cm/s',
                                     resolution=np.nan,
                                     timestamps=H5DataIO(timestamps, compression="gzip"))
            view_angle_obj.add_spatial_series(
                SpatialSeries(
                    name="SpatialSeries",
                    data=H5DataIO(view_angle_data, compression="gzip"),
                    reference_frame="unknown",
                    resolution=np.nan,
                    timestamps=H5DataIO(timestamps, compression="gzip")
                )
            )
            collision_ts = TimeSeries(name='Collision',
                                      data=H5DataIO(collision, compression="gzip"),
                                      unit='bool',
                                      resolution=np.nan,
                                      timestamps=H5DataIO(timestamps, compression="gzip"),
                                      description='boolean to indicate for each frame'
                                                  ' whether collision was detected')
            behavioral_processing_module = check_module(nwbfile, 'behavior',
                                                        'contains processed behavioral data')
            behavioral_processing_module.add_data_interface(pos_obj)
            behavioral_processing_module.add_data_interface(velocity_ts)
            behavioral_processing_module.add_data_interface(view_angle_obj)
            behavioral_processing_module.add_data_interface(time)
            behavioral_processing_module.add_data_interface(collision_ts)