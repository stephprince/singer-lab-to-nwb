import warnings

from datetime import datetime
from nwb_conversion_tools import NWBConverter, SpikeGadgetsRecordingInterface, PhySortingInterface
from pathlib import Path

from .update_task_data_interface import UpdateTaskVirmenInterface
from .mat_conversion_utils import convert_mat_file_to_dict


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """

    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        SpikeGadgetsRecording=SpikeGadgetsRecordingInterface,
        PhySorting=PhySortingInterface,
    )

    def __init__(self):

        """
               Initialize the NWBConverter object.
               """
        super().__init__(source_data=source_data)

    def get_metadata(self):

        virmen_file_path = Path(self.data_interface_objects['VirmenData'].source_data['file_path'])
        session_id = virmen_file_path.stem

        metadata = super().get_metadata()
        metadata['NWBFile'].update(
            experimenter='Steph Prince',
            session_id=session_id,
            institution="Georgia Tech",
            lab="Singer",
            session_description="Mice were head-fixed and performed the update task in virtual reality while neural "
                                "activity was recorded",
        )

        if virmen_file_path.is_file():
            session_data = convert_mat_file_to_dict(mat_file_name=virmen_file_path)
            subject_data = session_data['virmenData']['sessioninfo']
            metadata.update(
                Subject=dict(
                    subject_id=subject_data,
                    species="Mus Musculus",
                    genotype="Wild type, C57BL/6J",
                    sex="Male",
                    age="3-7 months",  # TODO check these ages
                )
            )
        else:
            warnings.warn(f"Warning: no subject file detected for session {session_id}!")

        metadata["Ecephys"]["Device"][0].update(description="64 channel, two-shank NeuroNexus Takahashi probes were "
                                                            "inserted into dorsal hippocampus (CA1) and medial "
                                                            "prefrontal cortex (prelimbic, infralimbic) each "
                                                            "recording session.")
