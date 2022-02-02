import warnings

from nwb_conversion_tools import NWBConverter, PhySortingInterface, CellExplorerSortingInterface
from pathlib import Path

from update_task_data_interface import UpdateTaskVirmenInterface
from mat_conversion_utils import convert_mat_file_to_dict


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """

    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        PhySortingCA1=PhySortingInterface,
        PhySortingPFC=PhySortingInterface
        #SpikeGadgetsRecording=SpikeGadgetsRecordingInterface,
        #CellExplorerSorting=CellExplorerSortingInterface,
    )

    def __init__(self,source_data):

        """
               Initialize the NWBConverter object.
               """
        super().__init__(source_data=source_data)

    def get_metadata(self):
        virmen_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = virmen_file_path.parent.stem

        metadata = super().get_metadata()
        metadata['NWBFile'].update(
            experimenter=["Steph Prince"],
            session_id=session_id,
            institution="Georgia Tech",
            lab="Singer",
            session_description="Head-fixed mice performed update task in virtual reality",
        )

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

        #if "Ecephys" not in metadata:
        #    metadata.update(Ecephys=SpikeGadgetsRecordingInterface.get_ecephys_metadata(xml_file_path=xml_file_path))

        # metadata["Ecephys"]["UnitProperties"] = [dict(name='Amplitude', description='amplitude of spike'),
        #                                          dict(name='ContamPct', description='proportion spikes within ISI'),
        #                                          dict(name='KSLabel', description='auto-kilosort label (pre-curation)')
        # ]

        return metadata