import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from pynwb import NWBHDF5IO

from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths, get_session_info
from mat_conversion_utils import matlab_time_to_datetime


# set up fixtures
@pytest.fixture(scope="module")
def file_paths():
    base_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data/")
    return get_file_paths(base_path, "S25_210913")


@pytest.fixture(scope="module")
def session_info():
    spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
    all_session_info = get_session_info(filename=spreadsheet_filename, animals=[25], dates_included=[210913])
    unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

    return unique_sessions.get_group(('S', 25, 210913))


@pytest.fixture(scope="module")
def nwbfile(file_paths, session_info):
    nwbfilename = Path(file_paths["nwbfile"]).with_name('virmen_testing.nwb')
    source_data = dict(VirmenData=dict(file_path=str(file_paths["virmen"]),
                                       session_id=file_paths["session_id"],
                                       synced_file_path=str(file_paths["processed_ephys"]),
                                       session_info=session_info))
    converter = SingerLabNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    converter.run_conversion(nwbfile_path=str(nwbfilename), metadata=metadata, overwrite=True)

    io = NWBHDF5IO(str(nwbfilename), 'r')
    nwbfile = io.read()

    return nwbfile


@pytest.fixture(scope="module")
def virmen_df_list(file_paths):
    virmen_files = list(file_paths["processed_ephys"].glob(f'virmenDataSynced*.csv'))
    virmen_df_list = []
    for files in virmen_files:
        virmen_df = pd.read_csv(files)
        virmen_df.dropna(subset=['spikeGadgetsTimes'], inplace=True)  # remove data w/o corresponding ephys data
        virmen_df_list.append(virmen_df)

    return virmen_df_list


@pytest.fixture(scope="module")
def virmen_df(virmen_df_list):
    return pd.concat(virmen_df_list, axis=0, ignore_index=True)


# test trial table conversion
def test_trial_conversion(nwbfile, virmen_df_list, virmen_df):
    # get different trial types from nwb file
    nwb_trials_df = nwbfile.trials.to_dataframe()
    nwb_num_trials = len(nwb_trials_df)
    nwb_num_left_trials = sum(nwb_trials_df['turn_type'] == 1)
    nwb_num_right_trials = sum(nwb_trials_df['turn_type'] == 2)
    nwb_num_update_trials = sum(nwb_trials_df['update_type'] == 2)
    nwb_num_stay_trials = sum(nwb_trials_df['update_type'] == 3)
    nwb_num_warmup_trials = sum(nwb_trials_df['maze_id'] == 3)
    nwb_num_correct_trials = sum(nwb_trials_df['correct'])
    nwb_duration_trials = round(sum(nwb_trials_df['duration']))  # round bc differences in iterations vs spikegadgets

    # get different  trial types from virmen file
    t_ends = np.array(virmen_df.index[virmen_df['taskState'] == 7]).squeeze()
    t_starts = np.array(
        virmen_df.index[virmen_df['taskState'] == 1]).squeeze() + 1  # add one for when teleporting actually occurs
    file_switch_inds = [len(df) for df in virmen_df_list]
    start_ind = 0
    trial_starts = []
    trial_ends = []
    for switch_ind in file_switch_inds:
        t_starts_temp = t_starts[(t_starts > start_ind) & (t_starts < start_ind + switch_ind)]
        t_ends_temp = t_ends[(t_ends > start_ind) & (t_ends < start_ind + switch_ind)]
        if t_starts_temp[0] > t_ends_temp[0]:  # if 1st trial end time is before the first start, remove first end
            t_ends_temp = t_ends_temp[1:]
        if t_ends_temp[-1] < t_starts_temp[-1]:  # if last trial start is after trial_end, remove last trial start
            t_starts_temp = t_starts_temp[:-1]
        start_ind = start_ind + switch_ind  # add file iteration lengths
        trial_starts.extend(t_starts_temp)
        trial_ends.extend(t_ends_temp)

    virmen_num_trials = len(trial_starts)
    virmen_num_left_trials = sum(virmen_df['trialType'][trial_starts] == 1)
    virmen_num_right_trials = sum(virmen_df['trialType'][trial_starts] == 2)
    virmen_num_update_trials = sum(virmen_df['trialTypeUpdate'][trial_starts] == 2)
    virmen_num_stay_trials = sum(virmen_df['trialTypeUpdate'][trial_starts] == 3)
    virmen_num_warmup_trials = sum(virmen_df['currentWorld'][trial_starts] == 3)
    virmen_num_correct_trials = sum(virmen_df['choice'][trial_ends] == 1)

    times = virmen_df['time'].apply(lambda x: matlab_time_to_datetime(x))
    times_starts = [(t - times[0]).total_seconds() for t in np.array(times)[trial_starts]]
    times_ends = [(t - times[0]).total_seconds() for t in np.array(times)[trial_ends]]
    durations = np.array(times_ends) - np.array(times_starts)
    virmen_duration_trials = round(sum(durations))

    # check that all values match up between groups
    assert nwb_num_trials == virmen_num_trials
    assert nwb_num_left_trials == virmen_num_left_trials
    assert nwb_num_right_trials == virmen_num_right_trials
    assert nwb_num_update_trials == virmen_num_update_trials
    assert nwb_num_stay_trials == virmen_num_stay_trials
    assert nwb_num_warmup_trials == virmen_num_warmup_trials
    assert nwb_num_correct_trials == virmen_num_correct_trials
    assert nwb_duration_trials == virmen_duration_trials


# test session durations
def test_session_duration(nwbfile, virmen_df_list, virmen_df):
    # get duration of individual behavioral intervals
    behavior_epochs = nwbfile.intervals['behavior_epochs'].to_dataframe()
    behavior_durations = behavior_epochs['stop_time'] - behavior_epochs['start_time']

    # get duration of individual virmen files
    virmen_durations = []
    for df in virmen_df_list:
        times = np.array(df['time'].apply(lambda x: matlab_time_to_datetime(x)))
        df_duration = (times[-1] - times[0]) / np.timedelta64(1, 's')
        virmen_durations.append(df_duration)

    # test that they match (round for small spikegadgets iteration differences)
    for nwb_dur, vr_dur in zip(behavior_durations, virmen_durations):
        assert round(nwb_dur) == round(vr_dur)


# def test that number of behavior files matches number on spreadsheet, what happens if no filess
def test_num_behavior_sessions(nwbfile, virmen_df_list, session_info):
    session_info_recs = np.where(session_info['Behavior'] == 1)[0] + 1  # add 1 for 1-based indexing
    assert len(session_info_recs) == len(virmen_df_list)


# def test number of reward events between files
def test_reward_events(nwbfile, virmen_df):
    nwb_reward_inds = np.where(nwbfile.processing['behavior']['rewards']['rewards'].data[:] == 1)[0]

    virmen_reward_info = np.diff(virmen_df['numRewards'], prepend=virmen_df['numRewards'][0])
    virmen_reward_inds = np.where(virmen_reward_info > 0)[0]

    assert len(nwb_reward_inds) == len(virmen_reward_inds)

