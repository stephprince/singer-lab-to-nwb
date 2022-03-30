import numpy as np
import pandas as pd
import shutil
import warnings

from pathlib import Path


class PhyUnitIDGenerator:
    """
    Generates new phy ids depending on the files given
    """
    def __init__(self, base_path, brain_regions, probe_channels):

        self.base_path = base_path
        self.brain_regions = brain_regions
        self.probe_channels = probe_channels

    def update_ids(self):
        num_units = 0
        num_channels = 0
        for br in self.brain_regions:
            # remap unit ids
            phy_folder = self.base_path / br / "sorted" / "kilosort"
            self.remap_ids(phy_folder, start_id=num_units, start_chan=num_channels)

            # update units and channel numbers for next region
            num_units = num_units + self.get_number_of_units(phy_folder)
            num_channels = num_channels + self.probe_channels

    def get_number_of_units(self, phy_folder):
        if phy_folder.is_dir():
            spike_clusters = np.load(Path(phy_folder, 'spike_clusters.npy'))
        else:
            spike_clusters = []
        return len(np.unique(spike_clusters))

    def remap_ids(self, phy_folder, start_id=0, start_chan=0):
        # make output path name
        phy_folder = Path(phy_folder)
        phy_folder_new = phy_folder / 'new_unit_ids'
        if phy_folder.is_dir():
            Path(phy_folder_new).mkdir(exist_ok=True)

            # copy files to new folder (and make backups for ones we'll change)
            shutil.copyfile((phy_folder / 'spike_times.npy'), (phy_folder_new / 'spike_times.npy'))
            shutil.copyfile((phy_folder / 'spike_templates.npy'), (phy_folder_new / 'spike_templates.npy'))
            shutil.copyfile((phy_folder / 'amplitudes.npy'), (phy_folder_new / 'amplitudes.npy'))
            shutil.copyfile((phy_folder / 'params.py'), (phy_folder_new / 'params.py'))
            shutil.copyfile((phy_folder / 'spike_clusters.npy'), (phy_folder / 'spike_clusters_backup.npy'))

            tsv_files = [x for x in phy_folder.iterdir() if (x.suffix == '.tsv') and ('backup' not in x.name)]
            for file in tsv_files:
                shutil.copyfile(file, phy_folder / f'{file.stem}_backup.tsv')

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
            assert len(np.unique(spike_clusters_new)) == len(np.unique(spike_clusters))
            assert spike_clusters_new[-1] == clust_id_map[spike_clusters[-1]]
            np.save((phy_folder_new / 'spike_clusters.npy'), spike_clusters_new)

            for file in tsv_files:
                clust_in = pd.read_csv(file, sep='\t')

                clust_out = clust_in.copy()
                if 'cluster_id' in clust_out.columns:
                    clust_out['cluster_id'].replace(clust_id_map, inplace=True)
                elif 'id' in clust_out.columns:
                    clust_out['id'].replace(clust_id_map, inplace=True)

                # if we used the phy structures to get the main channel, we would have to update them in the data structures
                # we're not doing this bc we don't use phy structures to get main channel currently and it might disrupt the
                # Cell Explorer process
                # if 'ch' in clust_out.columns:
                #     clust_out['ch'] = clust_out['ch'] + start_chan

                assert len(clust_in) == len(clust_out)

                # save new structures in new folder
                filename_new = phy_folder_new / file.name
                clust_out.to_csv(filename_new, sep='\t', index=False)

        else:
            warnings.warn(f'{phy_folder} not a valid directory. Skipping...')

