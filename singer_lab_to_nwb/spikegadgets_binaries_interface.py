import glob
import numpy as np
import pandas as pd

from pathlib import Path
from hdmf.backends.hdf5.h5_utils import H5DataIO

from pynwb import NWBFile
from pynwb.device import Device
from pynwb.ecephys import ElectrodeGroup, ElectricalSeries
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema, get_schema_from_hdmf_class
from readTrodesExtractedDataFile3 import readTrodesExtractedDataFile

from update_task_conversion_utils import get_spikegadgets_electrode_metadata
from array_iterators import MultiFileSpikeGadgetsIterator


class SpikeGadgetsBinariesInterface(BaseDataInterface):
    """
 Data interface for raw data from SpikeGadgets
 """

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        source_schema = dict(
            required=['raw_data_folder', 'session_info'],
            properties=dict(
                file_path=dict(type='string'),
                channel_map_path=dict(type='string'),
            ),
        )
        return source_schema

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
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

        # add ecephys info
        if "Ecephys" not in metadata:
            brain_regions = self.source_data['session_info'][['RegAB', 'RegCD']].values[0]
            channel_groups = range(len(brain_regions))
            sg_electrode_metadata = get_spikegadgets_electrode_metadata(brain_regions, channel_groups)

            metadata["Ecephys"] = sg_electrode_metadata

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):
        """Primary conversion function for the custom Singer lab behavioral interface."""
        # get general session details
        session_info = self.source_data['session_info']
        brain_regions = session_info[['RegAB', 'RegCD']].values[0]
        rec_files = session_info['Recording'].to_list()
        probe_file_path = Path(self.source_data['channel_map_path'])
        probe_map = pd.read_csv(probe_file_path)

        # get recording files
        raw_data_folder = Path(self.source_data['raw_data_folder'])
        num_channels = len(probe_map) * len(brain_regions)
        raw_filenames = []
        for rec in rec_files:  # get channel dirs ordered by depth first brain region and then second
            for ntrode in range(num_channels):
                raw_filenames.append(raw_data_folder / f'recording{rec}.raw/recording{rec}.raw_nt{ntrode+1}ch1.dat')
        assert len(raw_filenames) == len(rec_files) * num_channels, "Number of files does not match expected number"

        # add electrode info
        if not hasattr(nwbfile, 'electrode_groups'):
            num_electrodes = 0
            for group in metadata['Ecephys']['ElectrodeGroup']:
                # add device and electrode group if needed
                device_descript = [d['description'] for d in metadata['Ecephys']['Device'] if
                                   d['name'] == group['device']]
                if group['device'] not in nwbfile.devices:
                    device = nwbfile.create_device(name=group['device'], description=str(device_descript[0]))
                nwbfile.create_electrode_group(name=group['name'],
                                               description=group['description'],
                                               location=group['location'],
                                               device=device)

                # generate electrodes from probe
                if group['name'] is not 'analog_inputs':
                    for index, row in probe_map.iterrows():
                        nwbfile.add_electrode(id=index + num_electrodes,
                                              x=np.nan, y=np.nan, z=np.nan,
                                              rel_x=row['X'], rel_y=row['Y'], rel_z=row['K'],
                                              reference='ground pellet',
                                              imp=np.nan,
                                              location=group['location'],
                                              filtering='none',
                                              group=nwbfile.electrode_groups[group['name']])

                    num_electrodes = index + 1  # add 1 so that next loop creates new unique index

        elec_table_region = nwbfile.create_electrode_table_region(region=list(range(num_channels)),
                                                                  description='neuronexus_probes')

        # extract raw signals
        raw_obj = get_raw_timeseries(raw_data_folder, raw_filenames, rec_files, elec_table_region)
        nwbfile.add_acquisition(raw_obj)


def get_raw_timeseries(raw_data_folder, filenames, rec_files, elec_table_region):
    # get metadata from example files
    metadata = readTrodesExtractedDataFile(filenames[0])
    comments = f'original filename: {metadata["original_file"]}, ' \
               f'voltage_scaling: {metadata["voltage_scaling"]}, ' \
               f'trodes_version: {metadata["trodes_version"]},' \
               f'conversion info: "conversion field accounts for voltage gain/scaling and uV to V conversion factor"'
    samp_rate = float(metadata['clockrate'])
    conversion = round(float(metadata['voltage_scaling'])*0.001, 10)  # data in uV so multiply by 0.001 to get base units of volt

    # create data iterator for channels x recordings
    n_samp_per_rec = get_recording_n_samples(raw_data_folder, rec_files)
    n_channels = len(elec_table_region)
    data = MultiFileSpikeGadgetsIterator(filenames, readTrodesExtractedDataFile, rec_files, n_channels, n_samp_per_rec)

    # add electrical series to NWB file
    ecephys_ts = ElectricalSeries(name='raw_ecephys',
                                  data=H5DataIO(data, compression='gzip'),
                                  starting_time=0.0,
                                  rate=samp_rate,
                                  electrodes=elec_table_region,
                                  description='Raw unfiltered data acquired with spikegadgets acquisition system',
                                  conversion=conversion,
                                  comments=comments
                                  )
    return ecephys_ts


def get_recording_n_samples(data_folder, rec_files):
    raw_filenames = list(data_folder.glob(f'recording{rec_files}.raw/recording{rec_files}.raw_nt1ch1.dat'))
    durations = []
    for file in raw_filenames:
        raw_data = readTrodesExtractedDataFile(file)
        durations.append(len(raw_data['data']))  # duration in samples

    return durations
