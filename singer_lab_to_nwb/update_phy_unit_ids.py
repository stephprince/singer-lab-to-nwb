from phy_unit_id_generator import PhyUnitIDGenerator
from update_task_conversion_utils import get_file_paths

# set inputs
animals = [17, 20, 25, 28, 29]
dates_included = [210913]
dates_excluded = []
probe_channels = 64

# load session info
spreadsheet_filename = 'Y:/singer/Steph/Code/update-project/docs/metadata-summaries/VRUpdateTaskEphysSummary.csv'
all_session_info = get_session_info(filename=spreadsheet_filename, animals=animals,
                                dates_included=dates_included, dates_excluded=dates_excluded)
unique_sessions = all_session_info.groupby(['ID', 'Animal', 'Date'])

# loop through sessions and run conversion
for name, session in unique_sessions:

    # get session-specific info
    session_id = f"{name[0]}{name[1]}_{name[2]}"  # {ID}{Animal}_{Date} e.g. S25_210913
    brain_regions = session[['RegAB', 'RegCD']].values[0]
    file_paths = get_file_paths(session_id)

    # convert phy ids for each session
    phy_id_generator = PhyUnitIDGenerator(base_path=file_paths['processed_ephys_path'],
                                          brain_regions=brain_regions,
                                          probe_channels=probe_channels)
    phy_id_generator.update_ids()
