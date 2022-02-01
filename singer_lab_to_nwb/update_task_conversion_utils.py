from datetime import timedelta, datetime
from pathlib import Path


def get_file_paths(session_id, rec_id):
    """
    sets default file paths for different data types
    """

    base_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data")
    raw_ephys_path = base_path / "RawData" / "UpdateTask" / session_id
    processed_ephys_path = base_path / "ProcessedData" / "UpdateTask" / session_id
    virmen_path = base_path / "Virmen Logs" / "UpdateTask" / f"{session_id}_{rec_id}" / "virmenDataRaw.mat"
    kilosort_path = processed_ephys_path / "CA1" / "sorted" / "kilosort"
    cell_explorer_path = processed_ephys_path / "cell_type_classification.mat"
    probe_path = base_path / "ProbeData" / "update-task-64-chan-dual-probes.prb"
    nwbfile_path = base_path / "NWBFile" / f"{session_id}.nwb"

    return dict(raw_ephys=raw_ephys_path,
                processed_ephys=processed_ephys_path,
                virmen=virmen_path,
                cell_explorer=cell_explorer_path,
                probe=probe_path,
                nwbfile=nwbfile_path
                )

def matlab_time_to_datetime(series):
    times = datetime.fromordinal(int(series)) + \
            timedelta(days=series % 1) - \
            timedelta(days=366)
    return times

def check_module(nwbfile, name, description=None):
    """
    Check if processing module exists. If not, create it. Then return module.
    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)
    Returns
    -------
    pynwb.module
    """
    if name in nwbfile.modules:
        return nwbfile.modules[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)