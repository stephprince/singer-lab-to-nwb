from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths, get_session_info

# set inputs
animals = [17, 20, 25, 28, 29]
dates_included = [210913]
dates_excluded = []

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

    # add source data
    stub_test = False
    source_data = dict(
        VirmenData=dict(file_path=str(file_paths["virmen"]),
                        session_id=file_paths["session_id"],
                        synced_file_path=str(file_paths["processed_ephys"])),
        PreprocessedData=dict(processed_data_folder=str(file_paths["processed_ephys"]),
                              raw_data_folder=str(file_paths['raw_ephys']),
                              channel_map_path=str(file_paths["channel_map"]),
                              session_info=session))

    for br in brain_regions:
        phy_path = file_paths["processed_ephys"] / br / file_paths["kilosort"]
        source_data[f'PhySorting{br}'] = dict(folder_path=str(phy_path), exclude_cluster_groups=["noise", "mua"])

    # run the conversion process
    converter = SingerLabNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    converter.run_conversion(nwbfile_path=file_paths["nwbfile"], metadata=metadata, overwrite=True)
