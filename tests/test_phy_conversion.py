import pytest
import pandas as pd

from pathlib import Path
from pynwb import NWBHDF5IO

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
def brain_regions(session_info):
    return session_info[['RegAB', 'RegCD']].values[0]


@pytest.fixture(scope="module")
def nwbfile(file_paths, brain_regions):
    nwbfilename = Path(file_paths["nwbfile"]).with_name('phy_testing.nwb')

    # load source data for different brain regions
    source_data = dict()
    conversion_options = dict()
    for br in brain_regions:
        phy_path = file_paths["processed_ephys"] / br / file_paths["kilosort"]
        source_data[f'PhySorting{br}'] = dict(folder_path=str(phy_path), exclude_cluster_groups=["noise", "mua"])
        conversion_options[f'PhySorting{br}'] = dict(stub_test=True)

    # run conversion
    converter = SingerLabNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    converter.run_conversion(nwbfile_path=str(nwbfilename),
                             metadata=metadata,
                             conversion_options=conversion_options,
                             overwrite=True)

    io = NWBHDF5IO(str(nwbfilename), 'r')
    nwbfile = io.read()

    return nwbfile


def test_unit_numbers(nwbfile, file_paths, brain_regions):
    # load nwb file units table
    units_df = nwbfile.units.to_dataframe()

    # loop through brain regions and compare to phy files
    phy_num_units = []
    for br in brain_regions:
        # load phy information
        cluster_info_filename = file_paths['processed_ephys'] / br / file_paths['kilosort'] / 'cluster_info.tsv'
        clust_in = pd.read_csv(cluster_info_filename, sep='\t')

        phy_num_units.append(len(clust_in[clust_in['group'] == 'good']))

    assert sum(phy_num_units) == len(units_df)
