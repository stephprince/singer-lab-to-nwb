import pytest
import numpy as np
import pandas as pd

from pathlib import Path
from pynwb import NWBHDF5IO

from singer_lab_mat_loader import SingerLabMatLoader
from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths, get_session_info

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
    nwbfilename = Path(file_paths["nwbfile"]).with_name('preprocessing_testing.nwb')
    source_data = dict(PreprocessedData=dict(processed_data_folder=str(file_paths["processed_ephys"]),
                                             raw_data_folder=str(file_paths['raw_ephys']),
                                             channel_map_path=str(file_paths["channel_map"]),
                                             session_info=session_info))
    conversion_options = dict(PreprocessedData=dict(stub_test=True))  # stub test (shortest recording) for testing
    converter = SingerLabNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    converter.run_conversion(nwbfile_path=str(nwbfilename),
                             metadata=metadata,
                             conversion_options=conversion_options,
                             overwrite=True)

    io = NWBHDF5IO(str(nwbfilename), 'r')
    nwbfile = io.read()

    return nwbfile


def test_channel_mapping(nwbfile, file_paths):
    # load probe map
    probe_map = pd.read_csv(file_paths['channel_map'])

    # load NWB file channel mapping
    electrode_df = nwbfile.electrodes.to_dataframe()
    probes = electrode_df.groupby('group_name')
    for name, pr in probes:
        if name != 'analog_inputs':
            assert pr['hardware_channel'].values[0] == probe_map['HW Channel'].values[0]


def test_lfp_event_times(nwbfile, file_paths, session_info):
    # load ripple channel
    electrode_df = nwbfile.electrodes.to_dataframe()
    hardware_ripple_channel = electrode_df['hardware_channel'][electrode_df['ripple_channel'] == 1].values[0]

    # load lfp events from mat file
    recording_df = nwbfile.intervals['recording_epochs'].to_dataframe()
    rec_file = recording_df['recording_number'][recording_df['stub_test_includes']].values[0]
    event_filename = list((file_paths['processed_ephys'] / 'CA1' / str(hardware_ripple_channel)).glob(f'thetas{rec_file}.mat'))

    mat_loader = SingerLabMatLoader(session_info['Animal'].values[0], session_info['Date'].values[0])
    matin = mat_loader.run_conversion(event_filename, rec_file, 'scipy')
    mat_num_theta_events = np.size(matin.startind)
    mat_theta_durations = matin.endtime - matin.starttime

    # load lfp events from nwb fil
    thetas_df = nwbfile.processing['ecephys']['thetas'].to_dataframe()
    nwb_num_theta_events = len(thetas_df)
    nwb_theta_durations = np.array(thetas_df['stop_time']) - np.array(thetas_df['start_time'])

    assert mat_num_theta_events == nwb_num_theta_events
    assert np.round(mat_theta_durations, 2) == np.round(nwb_theta_durations, 2)
    assert any(event < nwb_theta_durations for event in thetas_df['min_duration'])
