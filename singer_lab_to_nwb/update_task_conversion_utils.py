from pathlib import Path

def get_file_paths(session_id, rec_id):
    """
    sets default file paths for different data types
    """

    base_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data")
    raw_ephys_path = base_path / "RawData" / "UpdateTask" / session_id
    processed_ephys_path = base_path / "ProcessedData" / "UpdateTask" / session_id
    virmen_path = base_path / "Virmen Logs" / "UpdateTask"
    kilosort_path_CA1 = processed_ephys_path / "CA1" / "sorted" / "kilosort"
    kilosort_path_PFC = processed_ephys_path / "PFC" / "sorted" / "kilosort"
    #cell_explorer_path = kilosort_path_CA1 / f"{session_id}_CA1.spikes.cellinfo.mat"
    #probe_path = base_path / "ProbeData" / "update-task-64-chan-dual-probes.prb"
    channel_map_path = base_path / "ProbeData" / "A2x32-Poly5-10mm-20s-200-100-RigC-ProbeMap.csv"
    nwbfile_path = str(base_path / "NWBFile" / f"{session_id}.nwb")

    return dict(raw_ephys=raw_ephys_path,
                processed_ephys=processed_ephys_path,
                virmen=virmen_path,
                kilosort_CA1=kilosort_path_CA1,
                kilosort_PFC=kilosort_path_PFC,
                channel_map=channel_map_path,
                nwbfile=nwbfile_path
                )

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


