def get_file_paths(session_id):
    """
    sets default file paths for different data types
    """

    base_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data")
    raw_ephys_path = base_path / "RawData" / "UpdateTask" / session_id
    processed_ephys_path = base_path / "ProcessedData" / "UpdateTask" / session_id
    kilosort_paths = [processed_ephys_path / "CA1" / "sorted", processed_ephys_path / "PFC" / "sorted"]
    probe_path = base_path / "ProbeData" / "update-task-64-chan-dual-probes.prb"
    nwbfile_path = base_path / session_id

    return dict(virmen=virmen_path,
                raw_ephys=raw_ephys_path,
                processed_ephys=processed_ephys_path,
                kilosor=kilosort_paths,
                probe=probe_path,
                nwbfile=nwbfile_path
                )
