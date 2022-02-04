from pathlib import Path
from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths, update_phy_unit_ids

# get file paths for conversion
session_id = "S25_210913"
rec_id = "1"
brain_regions = ['CA1','PFC']
file_paths = get_file_paths(session_id, rec_id)

# run conversion processes
stub_test = True
#conversion_gain = [0.195]

source_data = dict(
    VirmenData=dict(file_path=str(file_paths["virmen"])),
    PreprocessedData=dict(processed_data_folder=str(file_paths["processed_ephys"]),
                          channel_map_path=str(file_paths["channel_map"]),
                          brain_regions=brain_regions),
    PhySortingCA1=dict(folder_path=str(file_paths["kilosort_CA1"]), exclude_cluster_groups=["noise", "mua"]),
    PhySortingPFC=dict(folder_path=str(file_paths["kilosort_PFC"]), exclude_cluster_groups=["noise", "mua"]),
    # SpikeGadgetsRecording=dict(
        #     filename=str(file_paths["raw_ephys"]),
        #     gains=conversion_gain,  # SpikeGadgets requires manual specification of the conversion factor
        #     probe_file_path=str(file_paths["probe"]),
        # ),
    #CellExplorerSorting=dict(spikes_matfile_path=str(file_paths["cell_explorer"])),
)

conversion_options = dict(
    PhySortingCA1=dict(stub_test=stub_test),
    PhySortingPFC=dict(stub_test=stub_test),
    #SpikeGadgetsRecording=dict(stub_test=stub_test),
    #CellExplorerSorting=dict(stub_test=stub_test)
)

# remake kilosort files to update unit ids
if 'PhySortingCA1' in source_data.keys(): # TODO - make this more generalizable for any phy sorting/number of regions
    update_phy_unit_ids(source_data)

# run the conversion process
converter = SingerLabNWBConverter(source_data=source_data)
metadata = converter.get_metadata()
converter.run_conversion(
    nwbfile_path=file_paths["nwbfile"],
    metadata=metadata,
    conversion_options=conversion_options,
    overwrite=True,
)