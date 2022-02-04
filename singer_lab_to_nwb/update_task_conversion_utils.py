import numpy as np
import pandas as pd
import shutil

from datetime import timedelta, datetime
from pathlib import Path


def get_file_paths(session_id, rec_id):
    """
    sets default file paths for different data types
    """

    base_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data")
    raw_ephys_path = base_path / "RawData" / "UpdateTask" / session_id / 'recording1_20210913_170611.rec'
    processed_ephys_path = base_path / "ProcessedData" / "UpdateTask" / session_id
    virmen_path = base_path / "Virmen Logs" / "UpdateTask" / f"{session_id}_{rec_id}" / "virmenDataRaw.mat"
    kilosort_path_CA1 = processed_ephys_path / "CA1" / "sorted" / "kilosort"
    kilosort_path_PFC = processed_ephys_path / "PFC" / "sorted" / "kilosort"
    #cell_explorer_path = kilosort_path_CA1 / f"{session_id}_CA1.spikes.cellinfo.mat"
    #probe_path = base_path / "ProbeData" / "update-task-64-chan-dual-probes.prb"
    channel_map_path = base_path / "ProbeData" / "A2x32-Poly5-10mm-20s-200-100-probemap.csv"
    nwbfile_path = str(base_path / "NWBFile" / f"{session_id}.nwb")

    return dict(raw_ephys=raw_ephys_path,
                processed_ephys=processed_ephys_path,
                virmen=virmen_path,
                kilosort_CA1=kilosort_path_CA1,
                kilosort_PFC=kilosort_path_PFC,
                channel_map=channel_map_path,
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


def update_phy_unit_ids(source_data):
    # update the unit ids
    remap_unit_ids(source_data['PhySortingCA1']['folder_path'], start_id=0)
    num_units = get_number_of_units(source_data['PhySortingCA1']['folder_path'])
    remap_unit_ids(source_data['PhySortingPFC']['folder_path'], start_id=num_units)

    #update the path pointed to by the source_data
    source_data['PhySortingCA1']['folder_path'] = str(Path(source_data['PhySortingCA1']['folder_path'], 'new_unit_ids'))
    source_data['PhySortingPFC']['folder_path'] = str(Path(source_data['PhySortingPFC']['folder_path'], 'new_unit_ids'))

def get_number_of_units(phy_folder):
    spike_clusters = np.load(Path(phy_folder, 'spike_clusters.npy'))
    return len(np.unique(spike_clusters))

def remap_unit_ids(phy_folder, start_id=0):
    # TODO - remap the channel numbers for the electrodes as well??
    # make output path name
    phy_folder = Path(phy_folder)
    phy_folder_new = phy_folder / 'new_unit_ids'
    Path(phy_folder_new).mkdir(exist_ok=True)

    # get new and old cluster ids
    spike_clusters = np.load(phy_folder / 'spike_clusters.npy')
    clust_id = np.unique(spike_clusters)
    end_id = start_id + len(clust_id)
    clust_id_new = range(start_id, end_id)
    clust_id_map = dict(zip(clust_id, clust_id_new))

    # replace cluster ids for relevant files
    spike_clusters_new = np.copy(spike_clusters)
    for key, value in clust_id_map.items():
        spike_clusters_new[spike_clusters == key] = value
    np.save((phy_folder_new / 'spike_clusters.npy'), spike_clusters_new)

    # copy spike times to new file # TODO: figure out which of these I can just copy later
    shutil.copyfile((phy_folder / 'spike_times.npy'), (phy_folder_new / 'spike_times.npy'))
    shutil.copyfile((phy_folder / 'spike_templates.npy'), (phy_folder_new / 'spike_templates.npy'))
    shutil.copyfile((phy_folder / 'params.py'), (phy_folder_new / 'params.py'))

    tsv_files = [x for x in phy_folder.iterdir() if x.suffix == '.tsv']
    for file in tsv_files:
        clust_in = pd.read_csv(file, sep='\t')

        clust_out = clust_in.copy()
        if 'cluster_id' in clust_out.columns:
            clust_out['cluster_id'].replace(clust_id_map, inplace=True)
        elif 'id' in clust_out.columns:
            clust_out['id'].replace(clust_id_map, inplace=True)

        # save new structures in new folder
        filename_new = phy_folder_new / file.name
        clust_out.to_csv(filename_new, sep='\t', index=False)