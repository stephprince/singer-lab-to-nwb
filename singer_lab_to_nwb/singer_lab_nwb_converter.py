import warnings

from nwb_conversion_tools import NWBConverter, PhySortingInterface, SpikeGadgetsRecordingInterface
from pathlib import Path

from update_task_virmen_interface import UpdateTaskVirmenInterface
from singer_lab_preprocessing_interface import SingerLabPreprocessingInterface
from spikegadgets_binaries_interface import SpikeGadgetsBinariesInterface
from mat_conversion_utils import convert_mat_file_to_dict


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """

    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        SpikeGadgetsData=SpikeGadgetsBinariesInterface,
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

        # add general experiment info
        virmen_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = virmen_file_path.parent.stem


        metadata['NWBFile'].update(
            experimenter=["Steph Prince"],
            session_id=session_id,
            institution="Georgia Tech",
            lab="Singer",
            session_description="Head-fixed mice performed update task in virtual reality",
        )

        # add virmen subject info
        if virmen_file_path:
            session_data = convert_mat_file_to_dict(mat_file_name=virmen_file_path)
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
        else:
            warnings.warn(f"Warning: no subject file detected for session {session_id}!")

        # add spike sorting column info
        sorted_path = Path(self.data_interface_objects['PhySortingCA1'].source_data['folder_path'])
        if sorted_path:
            metadata["Ecephys"]["UnitProperties"] = [dict(name='Amplitude', description='amplitude imported from phy'),
                                                     dict(name='ContamPct', description='contampct imported from phy'),
                                                     dict(name='KSLabel', description='auto-label (pre-curation)')
                                                     ]

        return metadata
