import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class
from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup

class SingerLabPreprocessingInterface(BaseDataInterface):
    """
    Data interface for preprocessed, filtered data in standard Singer Lab format

    Data is stored in
    """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        source_schema = dict(
            required=['processed_data_folder','channel_map_path'],
            properties=dict(
                file_path=dict(type='string'),
                channel_map_path=dict(type='string'),
            ),
        )
        return source_schema

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
        # TODO - figure out what all this schema setup does and if I need it
        metadata_schema["properties"]["Ecephys"] = get_base_schema(tag="Ecephys")
        metadata_schema["properties"]["Ecephys"]["required"] = ["Device", "ElectrodeGroup"]
        metadata_schema["properties"]["Ecephys"]["properties"] = dict(
            Device=dict(type="array", minItems=1, items={"$ref": "#/properties/Ecephys/properties/definitions/Device"}),
            ElectrodeGroup=dict(
                type="array", minItems=1, items={"$ref": "#/properties/Ecephys/properties/definitions/ElectrodeGroup"}
            ),
            Electrodes=dict(
                type="array",
                minItems=0,
                renderForm=False,
                items={"$ref": "#/properties/Ecephys/properties/definitions/Electrodes"},
            ),
        )
        # Schema definition for arrays
        metadata_schema["properties"]["Ecephys"]["properties"]["definitions"] = dict(
            Device=get_schema_from_hdmf_class(Device),
            ElectrodeGroup=get_schema_from_hdmf_class(ElectrodeGroup),
            Electrodes=dict(
                type="object",
                additionalProperties=False,
                required=["name"],
                properties=dict(
                    name=dict(type="string", description="name of this electrodes column"),
                    description=dict(type="string", description="description of this electrodes column"),
                ),
            ),
        )
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()

        # TODO - figure out if I want to load up device info here or in the main function, think the main function?
        # metadata["Ecephys"] = dict(
        #     Device=[dict(name="NeuroNexus probe", description="64-channel probe ")],
        #     ElectrodeGroup=[
        #         dict(name=str(group_id), description="no description", location="unknown", device="NeuroNexus probe")
        #         for group_id in np.unique(self.recording_extractor.get_channel_groups())
        #     ],
        # )
        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # load probe and channel details
        processed_data_folder = self.source_data['processed_data_folder']
        probe_file_path = Path(self.source_data['channel_map_path'])
        df = pd.read_csv(probe_file_path)

        # extract recording start and end points - store as epochs
        # use the first channel of the first region for this

        # extract analog signals
        analog_signals = ['licks', 'rotVelocity', 'transVelocity'] # stored as acquisition time series?
        digital_signals = ['delay', 'trial', 'update'] # stored as behavioral events

        # extract digital signals


        # extract filtered lfp signals



        # if Path(mat_file).is_file():
        #     # convert the mat file into dict and data frame format
        #     matin = convert_mat_file_to_dict(mat_file)

