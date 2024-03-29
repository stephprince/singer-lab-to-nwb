import scipy.io
import numpy as np
import h5py

from pathlib import Path
from pynwb import NWBFile
from nwb_conversion_tools.basedatainterface import BaseDataInterface
from nwb_conversion_tools.utils.json_schema import get_base_schema

from mat_conversion_utils import convert_mat_file_to_dict


class CellExplorerCustomInterface(BaseDataInterface):
    """Primary data interface class for converting Cell Explorer spiking data."""

    def __init__(self, **source_data):
        super().__init__(**source_data)
        self.properties_dict = dict(brainRegion=dict(name='region',
                                                     description='brain region where each unit was detected'),
                                    probeID=dict(name='electrode_group',
                                                 description='electrode group unit was identified on'),
                                    shankID=dict(name='shank_id',
                                                 description='electrode shank unit was identified on'),
                                    maxWaveformCh=dict(name='main_channel',
                                                       description='0-based recording channel with largest waveform'),
                                    phy_maxWaveformCh1=dict(name='main_channel_phy',
                                                            description='0-based main recording channel from phy'),
                                    trilat_x=dict(name='location_on_electrode_x',
                                                  description='x coordinate in um of unit on probe'),
                                    trilat_y=dict(name='location_on_electrode_y',
                                                  description='y coordinate in um of unit on probe'),
                                    spikeCount=dict(name='spike_count',
                                                    description='Spike count for entire session, cell explorer removes '
                                                                'spikes within 0.5ms so this value will be smaller than'
                                                                ' the length of the phy spike_times column'),
                                    firingRate=dict(name='firing_rate',
                                                    description='Spike count normalized by interval btwn first and last spike'),
                                    firingRateInstability=dict(name='firing_rate_instability',
                                                               description='Mean of absolute differential firing rate across time'),
                                    refractoryPeriodViolation=dict(name='isi_violation',
                                                                   description='proportion of ISIs less than 2 ms long'),
                                    tags=dict(name='cell_explorer_label',
                                              description='classification as good or bad from cell explorer manual curation'),
                                    putativeCellType=dict(name='cell_type',
                                                          description='cell type classification'),
                                    troughToPeak=dict(name='spike_width',
                                                      description='trough to peak waveform width used for classification'),
                                    acg_tau_rise=dict(name='acg_tau_rise',
                                                      description='autocorrelogram metric used for classification'),
                                    ab_ratio=dict(name='ab_ratio',
                                                  description='waveform asymmetry, ratio between positive peaks'),
                                    peakVoltage=dict(name='spike_amplitude',
                                                     description='peak voltage from main channel (max-min of waveform)'),
                                    polarity=dict(name='polarity',
                                                  description='average voltage of spike vs baseline used to flip positive waveforms for metrics'),
                                    waveforms=dict(filt=dict(name='waveform_mean',
                                                             description='mean filtered waveform'),
                                                   filt_std=dict(name='waveform_sd',
                                                                 description='std filtered waveform'),
                                                   time=dict(name='waveform_time',
                                                            description='time values around the waveform peak'),
                                                   ),
                                    )

    @classmethod
    def get_source_schema(cls):
        """Compile input schemas from each of the data interface classes."""
        return dict(
            required=['file_path'],
            properties=dict(
                file_path=dict(type='string'),
            )
        )

    def get_metadata_schema(self):
        """Compile metadata schemas from each of the data interface objects."""
        metadata_schema = get_base_schema()
        return metadata_schema

    def get_metadata(self):
        metadata = super().get_metadata()

        # append unit properties to the metadata table
        unit_properties = []
        celltype_filepath = Path(self.source_data["file_path"])
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

    def run_conversion(self, nwbfile: NWBFile, metadata: dict, stub_test: bool = False, ):
        # load up session data
        session_info = self.source_data['session_info']
        brain_regions = session_info[['RegAB', 'RegCD']].values[0]
        celltype_filepath = Path(self.source_data["file_path"])
        celltype_data = convert_mat_file_to_dict(celltype_filepath)['cell_metrics']

        # sort data to match existing units table
        unit_ids = np.array(nwbfile.units['id'][:])
        unit_sort_index = np.argsort(np.argsort(unit_ids))  # get how to sort cell type index to match cluster order
        celltype_data_sorted = dict()
        for key, value in celltype_data.items():
            sorted_output = dict()
            if isinstance(value, np.ndarray):
                sorted_output = value[unit_sort_index]
            elif isinstance(value, dict) and key in ['responseCurves', 'waveforms', 'spikes']:
                for k, v in value.items():
                    sorted_output[k] = v[unit_sort_index]
            elif isinstance(value, dict) and key in ['acg', 'isi']:
                for k, v in value.items():
                    sorted_output[k] = v[:, unit_sort_index]
            elif isinstance(value, dict) and key in ['tags']:
                v = ['good' if unit in value['Good']-1 else 'bad' for unit in range(len(nwbfile.units))]
                where_bad = [ind+1 for ind, val in enumerate(v) if val == 'bad']  # adjust for 1-indexing
                if 'Bad' in value:  # if not empty
                    assert [value['Bad']] == where_bad[:], 'Unit tags incorrectly assigned'
                sorted_output = np.array(v)[unit_sort_index]
            celltype_data_sorted[key] = sorted_output

        assert all(celltype_data_sorted['cluID'] == unit_ids)

        # adjust channel and electrode data
        region_mapping = dict(zip(brain_regions, range(len(brain_regions))))
        channel_mapping = dict(zip(brain_regions, [0, 64]))
        probe_ids = [f'probe{region_mapping[r]}' for r in celltype_data_sorted['brainRegion']]
        chan_adjustment = [channel_mapping[r] for r in celltype_data_sorted['brainRegion']]
        celltype_data_sorted['probeID'] = np.array(probe_ids)
        celltype_data_sorted['maxWaveformCh'] = celltype_data_sorted['maxWaveformCh'] + chan_adjustment
        celltype_data_sorted['phy_maxWaveformCh1'] = celltype_data_sorted['phy_maxWaveformCh1'] + chan_adjustment - 1

        unit_col_args = dict()
        for name, prop_info in self.properties_dict.items():

            column_data = celltype_data_sorted.get(name, [])
            if name in ['waveforms']:
                for key, value in prop_info.items():
                    data = column_data.get(key, [])
                    data = data.tolist()
                    column_name = value['name']
                    column_description = value['description']

                    if column_name == 'waveform_time':
                        data = [np.diff(d)[0]/1000 for d in data]  # to get waveform rate in seconds from time vector
                        column_name = 'waveform_rate_cell_explorer'

                    unit_col_args = dict(name=column_name, description=column_description, data=data)
                    nwbfile.units.add_column(**unit_col_args)
            else:
                column_name = prop_info['name']
                column_description = prop_info['description']
                assert (len(column_data) == len(nwbfile.units))

                if isinstance(column_data[0], str):
                    column_data = column_data.tolist()

                if name in ['probeID'] and nwbfile.electrodes is not None:  # convert electrode string to
                    column_data = [nwbfile.electrode_groups[val] for val in column_data]

                unit_col_args = dict(name=column_name, description=column_description, data=column_data)
                nwbfile.units.add_column(**unit_col_args)

        # check spike counts match, won't work if a stub test bc short version of phy data
        if not stub_test:
            for ind in range(len(nwbfile.units)):
                phy_spike_count = len(nwbfile.units.get_unit_spike_times(ind))
                cell_explorer_spike_count = nwbfile.units['spike_count'][ind]
                assert (phy_spike_count-cell_explorer_spike_count)/phy_spike_count < 0.01  # < 1% difference, see note
                # NOTE: cell explorer removes spikes that are within 0.5 ms of each other.
                # so the cell explorer spike count will always be less than the phy spike count
                # this check exists more to verify that the units are being assigned correctly
