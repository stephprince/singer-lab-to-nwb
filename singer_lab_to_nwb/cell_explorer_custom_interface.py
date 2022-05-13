import scipy.io
import numpy as np

from pathlib import Path
from pynwb import NWBFile
from nwb_conversion_tools.basedatainterface import BaseDataInterface

from mat_conversion_utils import convert_mat_file_to_dict


class CellExplorerCustomInterface(BaseDataInterface):
    """Primary data interface class for converting Cell Explorer spiking data."""

    def get_metadata(self):
        metadata = super().get_metadata()

        self.properties_dict = dict(brainRegion=dict(name='region',
                                                     description='brain region where each unit was detected'),
                                    shankID=dict(name='group_id',
                                                 description='electrode group ID unit was identified on'),
                                    trilat_x=dict(name='location_on_electrode_x',
                                                  description='x coordinate in um of unit on probe'),
                                    trilat_y=dict(name='location_on_electrode_y',
                                                  description='y coordinate in um of unit on probe'),
                                    spikeCount=dict(name='spike_count',
                                                    description='Spike count for entire session'),
                                    firingRate=dict(name='firing_rate',
                                                    description='Spike count normalized by interval btwn first and last spike'),
                                    firingRateStd=dict(name='firing_rate_std',
                                                       description='standard deviation of firing rate across time'),
                                    firingRateInstability=dict(name='firing_rate_instability',
                                                               description='Mean of absolute differential firing rate across time'),
                                    refractoryPeriodViolation=dict(name='isi_violation',
                                                                   description='proportion of ISIs less than 2 ms long'),
                                    putativeCellType=dict(name='cell_type',
                                                          description='cell type classification'),
                                    troughToPeak=dict(name='spike_width',
                                                      description='trough to peak waveform width used for classification'),
                                    acg_tau_rise=dict(name='acg_tau_rise',
                                                      description='autocorrelogram metric used for classification'),
                                    acg=dict(name='autocorrelogram',
                                             description='autocorrelogram from -50 to +50 ms with 0.5 ms bins'),
                                    peakVoltage=dict(name='spike_amplitude',
                                                     description='peak voltage from main channel (max-min of waveform)'),
                                    polarity=dict(name='polarity',
                                                  description='average voltage of spike vs baseline used to flip positive waveforms for metrics'),
                                    waveforms=dict(waveform_filt=dict(name='waveform_filt_mean',
                                                                      description='mean filtered waveform'),
                                                   waveform_filt_std=dict(name='waveform_filt_std',
                                                                          description='std filtered waveform'),
                                                   waveform_raw=dict(name='waveform_raw_mean',
                                                                     description='mean raw waveform'),
                                                   waveform_raw_std=dict(name='waveform_raw_std',
                                                                         description='std raw waveform'),
                                                   ),
                                    maxWaveformCh=dict(name='max_channel',
                                                       description='0-based recording channel with largest waveform'),
                                    phy_maxWaveformCh1=dict(name='max_channel_phy',
                                                            description='1-based main recording channel from phy'),
                                    )

        # append unit properties to the metadata table
        unit_properties = []
        celltype_filepath = Path(self.source_data["cell_classification_path"])
        if celltype_filepath.is_file():
            celltype_info = scipy.io.loadmat(celltype_filepath).get("cell_metrics", np.empty(0))

            for key, value in self.properties_dict.items():
                if key in celltype_info.dtype.names:
                    if value.get('name'):
                        unit_properties.append(value)
                    else:
                        for k, v in value.items():
                            unit_properties.append(v)

        metadata.update(Ecephys=dict(UnitProperties=unit_properties))

        return metadata

    def run_conversion(self, nwbfile: NWBFile, metadata: dict):

        # load up the data
        celltype_filepath = Path(self.source_data["cell_classification_path"])
        celltype_data = convert_mat_file_to_dict(celltype_filepath)['cell_metrics']
        assert celltype_data['cluID'] == nwbfile.units[
            'id']  # TODO - why is my units table list appended out of order...
        # I think it's because of how the data interfaces are setup they don't like being out of order
        # Could just match the info to the cell ID using a sorting process

        unit_properties = metadata['Ecephys']['UnitProperties']
        unit_col_args = dict()
        for name, prop_info in self.properties_dict.items():  # TODO - figure out if better to loop through properties from metadata or self
            if name in ["max_channel", "max_electrode"] and nwbfile.electrodes is not None:
                unit_col_args.update(table=nwbfile.electrodes)

            column_data = celltype_data.get(name, [])
            if isinstance(column_data, dict):  # TODO - deal with this, could be recursive but leaving for now
                for key, value in prop_info.items():
                    column_data = column_data.get(key, [])
                    column_name = value['name']
                    column_description = value['description']

                    assert (len(column_data) == len(nwbfile.units))
                    unit_col_args = dict(name=column_name, description=column_description, data=column_data)
                    nwbfile.units.add_column(**unit_col_args)
            else:
                column_name = prop_info['name']
                column_description = prop_info['description']
                assert (len(column_data) == len(nwbfile.units))

                unit_col_args = dict(name=column_name, description=column_description, data=column_data)
                nwbfile.units.add_column(**unit_col_args)
