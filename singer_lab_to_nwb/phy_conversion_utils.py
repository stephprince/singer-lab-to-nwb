import numpy as np
import pandas as pd
import shutil

from pathlib import Path

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
    # make output path name  # TODO - change channel numbers to match first 64 and second 64
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

    # copy spike times to new file
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