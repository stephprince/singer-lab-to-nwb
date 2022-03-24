import pandas as pd

from git import Repo
from nwb_conversion_tools import NWBConverter, PhySortingInterface
from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
from pathlib import Path
from datetime import datetime, timedelta

from update_task_virmen_interface import UpdateTaskVirmenInterface
from singer_lab_preprocessing_interface import SingerLabPreprocessingInterface
from mat_conversion_utils import convert_mat_file_to_dict, matlab_time_to_datetime


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """
    # Note on the data interface classes:
    # If there's no source data provided, then the interfaces won't get used
    # So for the phy sorting, I could keep a bunch of brain regions here even if I don't use them all
    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        PreprocessedData=SingerLabPreprocessingInterface,
        PhySortingCA1=PhySortingInterface,
        PhySortingPFC=PhySortingInterface,  # I feel like there must be a better way to implement but leaving for now
    )

    def __init__(self, source_data):

        """
               Initialize the NWBConverter object.
               """
        super().__init__(source_data=source_data)

    def get_metadata(self):
        metadata = super().get_metadata()

        # get session info
        synced_file_path = self.data_interface_objects['VirmenData'].source_data['synced_file_path']
        synced_files = list(Path(synced_file_path).glob('virmenDataSynced*.csv'))
        if synced_files:
            session_info = self.data_interface_objects['VirmenData'].source_data['ephys_session_info']
            date_of_birth = datetime.strptime(str(session_info['DOB'].values[0]), '%y%m%d')
            date_of_recording = datetime.strptime(str(session_info['Date'].values[0]), '%y%m%d')
            age = f'{(date_of_recording - date_of_birth).days} days'
        else:  # if there's no ephys data, give approximate value (could add details later)
            age = '2-7 months'

        # add subject info
        session_id = self.data_interface_objects['VirmenData'].source_data['session_id']
        virmen_base_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        virmen_file_paths = list(virmen_base_path.glob(f'{session_id}*/virmenDataRaw.mat'))
        session_data = convert_mat_file_to_dict(mat_file_name=virmen_file_paths[0])
        virmen_df = pd.DataFrame(session_data['virmenData']['data'], columns=session_data['virmenData']['dataHeaders'])
        virmen_time = virmen_df["time"].apply(lambda x: matlab_time_to_datetime(x))
        subject_id = session_data['virmenData']["sessioninfo"]
        metadata.update(
            Subject=dict(
                subject_id=subject_id,
                species="Mus Musculus",
                genotype="Wild type, C57BL/6J",
                sex="Male",
                age=age,
            )
        )

        # get session start time
        if 'PreprocessedData' in self.data_interface_objects:  # if there exists ephys data, use that
            # load ephys timestamps file
            first_recording_file = session_info['Recording'].values[0]
            raw_ephys_folder = Path(self.data_interface_objects['PreprocessedData'].source_data['raw_data_folder'])
            timestamps_filename = list(raw_ephys_folder.glob(f'*.raw/*{first_recording_file}.timestamps.dat'))
            time_data = readTrodesExtractedDataFile(timestamps_filename[0])

            # extract session start time information
            file_creation_time = datetime.utcfromtimestamp(int(time_data['system_time_at_creation']) / 1e3)  # convert from ms to s
            samples_before_start = int(time_data['first_timestamp']) - int(time_data['timestamp_at_creation'])
            session_start_time = file_creation_time + timedelta(seconds=samples_before_start/int(time_data['clockrate']))
        else:  # otherwise use behavioral data
            session_start_time = virmen_time[0]
        assert session_start_time >= virmen_time[0], 'Invalid setup: behavior data occurs before session start time'

        # get git version
        repo = Repo(search_parent_directories=True)
        short_hash = repo.head.object.hexsha[:10]
        remote_url = repo.remotes.origin.url.strip('.git')

        # add general experiment info
        metadata['NWBFile'].update(
            experimenter=["Steph Prince"],
            session_id=session_id,
            institution="Georgia Tech",
            lab="Singer",
            session_description="Head-fixed mice performed update task in virtual reality",
            session_start_time=str(session_start_time),
            source_script=f'File created with git repo {remote_url}/tree/{short_hash}',
            source_script_file_name=f'convert_singer_lab_data.py',
        )

        # add spike sorting column info
        spike_sorting_data = any([key for key in self.data_interface_objects if 'PhySorting' in key])
        if spike_sorting_data:
            metadata["Ecephys"]["UnitProperties"] = [dict(name='Amplitude', description='amplitude imported from phy'),
                                                     dict(name='ContamPct', description='contampct imported from phy'),
                                                     dict(name='KSLabel', description='auto-label (pre-curation)'),
                                                     dict(name='ch', description='main channel of unit'),
                                                     dict(name='sh', description='shank of probe that unit is on'),]

        return metadata
