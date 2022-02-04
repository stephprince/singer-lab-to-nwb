import warnings

from nwb_conversion_tools import NWBConverter, PhySortingInterface, SpikeGadgetsRecordingInterface
from pathlib import Path

from update_task_virmen_interface import UpdateTaskVirmenInterface
from singer_lab_preprocessing_interface import SingerLabPreprocessingInterface
from mat_conversion_utils import convert_mat_file_to_dict


class SingerLabNWBConverter(NWBConverter):
    """
    Primary conversion class for Singer Lab data
    """

    data_interface_classes = dict(
        VirmenData=UpdateTaskVirmenInterface,
        PhySortingCA1=PhySortingInterface,
        PhySortingPFC=PhySortingInterface,
        PreprocessedData=SingerLabPreprocessingInterface,
        # CellExplorerSorting=CellExplorerSortingInterface,
    )

    def __init__(self, source_data):

        """
               Initialize the NWBConverter object.
               """
        super().__init__(source_data=source_data)

    def get_metadata(self):
        # add general experiment info
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

        # add ecephys info
        brain_regions = self.data_interface_objects['PreprocessedData'].source_data['brain_regions']
        brain_region_groups = [br for br in brain_regions for _ in range(2)] # double list so region for each shank
        channel_groups = range(len(brain_region_groups))

        spikegadgets = [dict(
            name="spikegadgets",
            description="Two NeuroNexus silicon probes with 2 (shanks) x 32 (channels) on each probe were inserted into"
                        " hippocampal CA1 and medial prefrontal cortex. Probes were 64-chan, poly5 Takahashi probe "
                        "formats. Electrophysiological data were acquired using a SpikeGadgets system digitized with 30"
                        " kHz rate."
        )
        ]

        electrode_group = [dict(
            name=f'shank{n+1}',
            description=f'shank{n+1} of NeuroNexus probes. Shanks 1-2 belong to probe 1, Shanks 3-4 belong to probe 2',
            location=brain_region_groups[n],
            device='spikegadgets')
            for n, _ in enumerate(channel_groups)
        ]

        metadata["Ecephys"] = dict(
            Device=spikegadgets,
            ElectrodeGroup=electrode_group,
            Electrodes=[
                dict(name="electrode_number",
                     description="0-indexed channel within the probe. Channels 0-31 belong to shank 1 and channels 32-64"
                                " belong to shank 2"),
                dict(name="group_name", description="Name of the electrode group this electrode is part of")
            ]
        )

        # add spike sorting column info
        metadata["Ecephys"]["UnitProperties"] = [dict(name='Amplitude', description='amplitude imported from phy'),
                                                 dict(name='ContamPct', description='contampct imported from phy'),
                                                 dict(name='KSLabel', description='auto-kilosort label (pre-curation)')
                                                 ]

        return metadata
