from pathlib import Path
from singer_lab_nwb_converter import SingerLabNWBConverter
from update_task_conversion_utils import get_file_paths

# get file paths for conversion
session_id = "S25_210913_1"
file_paths = get_file_paths(session_id)

# run conversion processes
stub_test = True
conversion_gain = [0.195]

source_data = dict(
    VirmenData=dict(file_path=str(file_paths["virmen"])),
    # SpikeGadgetsRecording=dict(
    #     filename=str(file_paths["raw_ephys"]),
    #     gains=conversion_gain,  # SpikeGadgets requires manual specification of the conversion factor
    #     probe_file_path=str(file_paths["probe"]),
    # ),
    #PhySorting=dict(folder_path=str(file_paths["kilosort"]), exclude_cluster_groups=["noise", "mua"])
)

conversion_options = dict(
    #SpikeGadgetsRecording=dict(stub_test=stub_test),
    #PhySorting=dict(stub_test=stub_test),
)

converter = SingerLabNWBConverter(source_data=source_data)
metadata = converter.get_metadata()
converter.run_conversion(
    nwbfile_path=file_paths["nwbfile"],
    metadata=metadata,
    conversion_options=conversion_options,
    overwrite=True,
)