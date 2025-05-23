from pathlib import Path

from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths, get_session_info

# set inputs
animals = [20]  # all animals: 17, 20, 25, 28, 29, 33, 34
dates_included = []
dates_excluded = []

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

    # add source data and conversion options
    stub_test = False
    skip_decomposition = False
    source_data = dict(
        VirmenData=dict(file_path=str(file_paths["virmen"]),
                        session_id=file_paths["session_id"],
                        synced_file_path=str(file_paths["processed_ephys"]),
                        session_info=session),
        PreprocessedData=dict(processed_data_folder=str(file_paths["processed_ephys"]),
                              raw_data_folder=str(file_paths['raw_ephys']),
                              channel_map_path=str(file_paths["channel_map"]),
                              session_info=session),
        SpikeGadgetsData=dict(raw_data_folder=str(file_paths['raw_ephys']),
                              channel_map_path=str(file_paths["channel_map"]),
                              session_info=session),
        CellExplorer=dict(file_path=str(file_paths['cell_explorer']),
                          session_info=session),)

    conversion_options = dict(PreprocessedData=dict(stub_test=stub_test,
                                                    skip_decomposition=skip_decomposition), )

    # add source data for brain region specific data
    for br in brain_regions:
        phy_path = file_paths["processed_ephys"] / br / file_paths["kilosort"]
        if phy_path.is_dir():
            source_data[f'PhySorting{br}'] = dict(folder_path=str(phy_path), exclude_cluster_groups=["noise", "mua"])
            conversion_options[f'PhySorting{br}'] = dict(stub_test=stub_test)

    # run the conversion process
    converter = SingerLabNWBConverter(source_data=source_data)
    metadata = converter.get_metadata()
    converter.run_conversion(nwbfile_path=file_paths["nwbfile"],
                             metadata=metadata,
                             conversion_options=conversion_options,
                             overwrite=True)
