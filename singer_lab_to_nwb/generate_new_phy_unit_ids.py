from pathlib import Path

from phy_unit_id_generator import PhyUnitIDGenerator
from update_task_conversion_utils import get_file_paths, get_session_info

# set inputs
animals = [33, 34]
dates_included = []
dates_excluded = [220422]
probe_channels = 64

# load session info
base_path = Path("Y:/singer")  # ALL file paths will be based on this base directory
spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# loop through sessions and run conversion
for name, session in unique_sessions:

    # get session-specific info
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    brain_regions = session[['RegAB', 'RegCD']].values[0]
    file_paths = get_file_paths(base_path, session_id)

    # convert phy ids for each session
    print(f'Updating ids for {session_id}')
    phy_id_generator = PhyUnitIDGenerator(base_path=file_paths['processed_ephys'],
                                          brain_regions=brain_regions,
                                          probe_channels=probe_channels)
    phy_id_generator.update_ids()
