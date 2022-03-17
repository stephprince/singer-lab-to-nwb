import warnings

from nwb_conversion_tools import NWBConverter, PhySortingInterface
from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile
from pathlib import Path
from datetime import datetime, timedelta

from update_task_virmen_interface import UpdateTaskVirmenInterface
from singer_lab_preprocessing_interface import SingerLabPreprocessingInterface
from mat_conversion_utils import convert_mat_file_to_dict


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """

    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        PreprocessedData=SingerLabPreprocessingInterface,
        PhySortingCA1=PhySortingInterface,
        PhySortingPFC=PhySortingInterface,
        # CellExplorerSorting=CellExplorerSortingInterface,
    )

    def __init__(self, source_data):

        """
               Initialize the NWBConverter object.
               """
        super().__init__(source_data=source_data)

    def get_metadata(self):
        metadata = super().get_metadata()

        # add subject info
        session_id = self.data_interface_objects['VirmenData'].source_data['session_id']
        virmen_base_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        virmen_file_paths = list(virmen_base_path.glob(f'{session_id}*/virmenDataRaw.mat'))
        session_data = convert_mat_file_to_dict(mat_file_name=virmen_file_paths[0])
        subject_id = session_data['virmenData']["sessioninfo"]
        metadata.update(
            Subject=dict(
                subject_id=subject_id,
                species="Mus Musculus",
                genotype="Wild type, C57BL/6J",
                sex="Male",
                age="3-7 months",  # TODO check these ages
            )
        )

        # get session start time
        if self.data_interface_objects['PreprocessedData']:  # if there exists ephys data, use that
            # load ephys timestamps file
            raw_ephys_folder = Path(self.data_interface_objects['PreprocessedData'].source_data['raw_data_folder'])
            timestamps_filename = list(raw_ephys_folder.glob(f'*.raw/*.timestamps.dat'))
            time_data = readTrodesExtractedDataFile(timestamps_filename[0])  # TODO - get first file from recording

            # extract session start time information
            file_creation_time = datetime.utcfromtimestamp(int(time_data['system_time_at_creation']) / 1e3)  # convert from ms to s
            samples_before_start = int(time_data['first_timestamp']) - int(time_data['timestamp_at_creation'])
            session_start_time = file_creation_time + timedelta(seconds=samples_before_start/int(time_data['clockrate']))
        else:  # otherwise use behavioral data
            virmen_df = pd.DataFrame(session_data['virmenData']['data'], columns=session_data['virmenData']['dataHeaders'])
            virmen_time = virmen_df["time"].apply(lambda x: matlab_time_to_datetime(x))
            session_start_time = virmen_time[0]

        # add general experiment info
        metadata['NWBFile'].update(
            experimenter=["Steph Prince"],
            session_id=session_id,
            institution="Georgia Tech",
            lab="Singer",
            session_description="Head-fixed mice performed update task in virtual reality",
            session_start_time=str(session_start_time),
        )
        # TODO - add tag for which git version this file was generated with

        # add spike sorting column info  # TODO - add fields that we want to save info for from phy
        sorted_path = Path(self.data_interface_objects['PhySortingCA1'].source_data['folder_path'])
        if sorted_path:
            metadata["Ecephys"]["UnitProperties"] = [dict(name='Amplitude', description='amplitude imported from phy'),
                                                     dict(name='ContamPct', description='contampct imported from phy'),
                                                     dict(name='KSLabel', description='auto-label (pre-curation)')
                                                     ]

        return metadata
