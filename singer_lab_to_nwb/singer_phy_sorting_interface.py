"""Authors: Heberto Mayorquin, Cody Baker."""

from typing import Union, Optional, List
import warnings
from warnings import warn
from collections import defaultdict

from spikeinterface.extractors import PhySortingExtractor
import spikeinterface as si
import spikeextractors as se
import numpy as np
from pynwb import NWBFile

import pynwb
from spikeinterface import BaseRecording, BaseSorting
from spikeinterface.core.old_api_utils import OldToNewSorting, OldToNewSorting
from spikeextractors import RecordingExtractor, SortingExtractor
from numbers import Real

from nwb_conversion_tools.datainterfaces.ecephys.basesortingextractorinterface import BaseSortingExtractorInterface
from nwb_conversion_tools.tools.nwb_helpers import get_module, make_or_load_nwbfile
from nwb_conversion_tools.utils import OptionalFilePathType, FolderPathType
from nwb_conversion_tools.tools.spikeinterface.spikeinterface import add_devices, add_electrode_groups, add_electrodes, set_dynamic_table_property, get_nspikes

SpikeInterfaceRecording = Union[BaseRecording, RecordingExtractor]
SpikeInterfaceSorting = Union[BaseSorting, SortingExtractor]


class SingerPhySortingInterface(BaseSortingExtractorInterface):
    """Primary data interface class for converting a PhySortingExtractor."""

    SX = PhySortingExtractor

    def __init__(
        self,
        folder_path: FolderPathType,
        exclude_cluster_groups: Optional[list] = None,
        verbose: bool = True,
        spikeextractors_backend: bool = False,
    ):
        if spikeextractors_backend:
            self.SX = se.PhySortingExtractor
        super().__init__(folder_path=folder_path, exclude_cluster_groups=exclude_cluster_groups, verbose=verbose)


    def run_conversion(
        self,
        nwbfile_path: OptionalFilePathType = None,
        nwbfile: Optional[NWBFile] = None,
        metadata: Optional[dict] = None,
        overwrite: bool = False,
        stub_test: bool = False,
        write_ecephys_metadata: bool = False,
        save_path: OptionalFilePathType = None,  # TODO: to be removed
    ):
        """
        Primary function for converting the data in a SortingExtractor to NWB format.

        Parameters
        ----------
        nwbfile_path: FilePathType
            Path for where to write or load (if overwrite=False) the NWBFile.
            If specified, the context will always write to this location.
        nwbfile: NWBFile, optional
            If passed, this function will fill the relevant fields within the NWBFile object.
            E.g., calling
                write_recording(recording=my_recording_extractor, nwbfile=my_nwbfile)
            will result in the appropriate changes to the my_nwbfile object.
            If neither 'save_path' nor 'nwbfile' are specified, an NWBFile object will be automatically generated
            and returned by the function.
        metadata: dict
            Information for constructing the nwb file (optional) and units table descriptions.
            Should be of the format::

                metadata["Ecephys"]["UnitProperties"] = dict(name=my_name, description=my_description)
        overwrite: bool, optional
            Whether or not to overwrite the NWBFile if one exists at the nwbfile_path.
            The default is False (append mode).
        stub_test: bool, optional (default False)
            If True, will truncate the data to run the conversion faster and take up less memory.
        write_ecephys_metadata: bool (optional, defaults to False)
            Write electrode information contained in the metadata.
        """
        if write_ecephys_metadata and "Ecephys" in metadata:
            n_channels = max([len(x["data"]) for x in metadata["Ecephys"]["Electrodes"]])
            recording = si.NumpyRecording(
                traces_list=[np.array(range(n_channels))],
                sampling_frequency=self.sorting_extractor.get_sampling_frequency(),
            )
            add_devices(recording=recording, nwbfile=nwbfile, metadata=metadata)
            add_electrode_groups(recording=recording, nwbfile=nwbfile, metadata=metadata)
            add_electrodes(recording=recording, nwbfile=nwbfile, metadata=metadata)
        if stub_test:
            sorting_extractor = self.subset_sorting()
        else:
            sorting_extractor = self.sorting_extractor
        property_descriptions = dict()
        for metadata_column in metadata.get("Ecephys", dict()).get("UnitProperties", []):
            property_descriptions.update({metadata_column["name"]: metadata_column["description"]})
            for unit_id in sorting_extractor.get_unit_ids():
                # Special condition for wrapping electrode group pointers to actual object ids rather than string names
                if metadata_column["name"] == "electrode_group":
                    if nwbfile.electrode_groups:
                        sorting_extractor.set_unit_property(
                            unit_id=unit_id,
                            property_name=metadata_column["name"],
                            value=nwbfile.electrode_groups[
                                self.sorting_extractor.get_unit_property(
                                    unit_id=unit_id, property_name="electrode_group"
                                )
                            ],
                        )
        custom_write_sorting(
            sorting_extractor,
            nwbfile_path=nwbfile_path,
            nwbfile=nwbfile,
            metadata=metadata,
            overwrite=overwrite,
            verbose=self.verbose,
            save_path=save_path,
            property_descriptions=property_descriptions,
        )

    
def custom_write_sorting(
    sorting: SortingExtractor,
    nwbfile_path: OptionalFilePathType = None,
    nwbfile: Optional[pynwb.NWBFile] = None,
    metadata: Optional[dict] = None,
    overwrite: bool = False,
    verbose: bool = True,
    property_descriptions: Optional[dict] = None,
    skip_properties: Optional[List[str]] = None,
    skip_features: Optional[List[str]] = None,
    write_as: str = "units",
    units_name: str = "units",
    units_description: str = "Autogenerated by nwb_conversion_tools.",
    save_path: OptionalFilePathType = None,  # TODO: to be removed
):
    """
    Primary method for writing a SortingExtractor object to an NWBFile.

    Parameters
    ----------
    sorting: SortingExtractor
    nwbfile_path: FilePathType
        Path for where to write or load (if overwrite=False) the NWBFile.
        If specified, the context will always write to this location.
    nwbfile: NWBFile, optional
        If passed, this function will fill the relevant fields within the NWBFile object.
        E.g., calling
            write_recording(recording=my_recording_extractor, nwbfile=my_nwbfile)
        will result in the appropriate changes to the my_nwbfile object.
        If neither 'save_path' nor 'nwbfile' are specified, an NWBFile object will be automatically generated
        and returned by the function.
    metadata: dict, optional
        Metadata dictionary with information used to create the NWBFile when one does not exist or overwrite=True.
    overwrite: bool, optional
        Whether or not to overwrite the NWBFile if one exists at the nwbfile_path.
        The default is False (append mode).
    verbose: bool, optional
        If 'nwbfile_path' is specified, informs user after a successful write operation.
        The default is True.
    property_descriptions: dict
        For each key in this dictionary which matches the name of a unit
        property in sorting, adds the value as a description to that
        custom unit column.
    skip_properties: list of str
        Each string in this list that matches a unit property will not be written to the NWBFile.
    skip_features: list of str
        Each string in this list that matches a spike feature will not be written to the NWBFile.
    write_as: str (optional, defaults to 'units')
        How to save the units table in the nwb file. Options:
        - 'units' will save it to the official NWBFile.Units position; recommended only for the final form of the data.
        - 'processing' will save it to the processing module to serve as a historical provenance for the official table.
    units_name : str (optional, defaults to 'units')
        The name of the units table. If write_as=='units', then units_name must also be 'units'.
    units_description : str (optional)
    """
    assert save_path is None or nwbfile is None, "Either pass a save_path location, or nwbfile object, but not both!"
    if nwbfile is not None:
        assert isinstance(nwbfile, pynwb.NWBFile), "'nwbfile' should be a pynwb.NWBFile object!"

    assert write_as in [
        "units",
        "processing",
    ], f"Argument write_as ({write_as}) should be one of 'units' or 'processing'!"
    if write_as == "units":
        assert units_name == "units", "When writing to the nwbfile.units table, the name of the table must be 'units'!"
    write_in_processing_module = False if write_as == "units" else True

    # TODO on or after August 1st, 2022, remove argument and deprecation warnings
    if save_path is not None:
        will_be_removed_str = "will be removed on or after August 1st, 2022. Please use 'nwbfile_path' instead."
        if nwbfile_path is not None:
            if save_path == nwbfile_path:
                warn(
                    "Passed both 'save_path' and 'nwbfile_path', but both are equivalent! "
                    f"'save_path' {will_be_removed_str}",
                    DeprecationWarning,
                )
            else:
                warn(
                    "Passed both 'save_path' and 'nwbfile_path' - using only the 'nwbfile_path'! "
                    f"'save_path' {will_be_removed_str}",
                    DeprecationWarning,
                )
        else:
            warn(
                f"The keyword argument 'save_path' to 'spikeinterface.write_recording' {will_be_removed_str}",
                DeprecationWarning,
            )
            nwbfile_path = save_path

    with make_or_load_nwbfile(
        nwbfile_path=nwbfile_path, nwbfile=nwbfile, metadata=metadata, overwrite=overwrite, verbose=verbose
    ) as nwbfile_out:
        custom_add_units_table(
            sorting=sorting,
            nwbfile=nwbfile_out,
            property_descriptions=property_descriptions,
            skip_properties=skip_properties,
            skip_features=skip_features,
            write_in_processing_module=write_in_processing_module,
            units_table_name=units_name,
            unit_table_description=units_description,
            write_waveforms=False,
        )
    return nwbfile_out

def custom_add_units_table(
    sorting: SpikeInterfaceSorting,
    nwbfile: pynwb.NWBFile,
    property_descriptions: Optional[dict] = None,
    skip_properties: Optional[List[str]] = None,
    skip_features: Optional[List[str]] = None,
    units_table_name: str = "units",
    unit_table_description: str = "Autogenerated by nwb_conversion_tools.",
    write_in_processing_module: bool = False,
    write_waveforms: bool = False,
):
    """
    Primary method for writing a SortingExtractor object to an NWBFile.

    Parameters
    ----------
    sorting: SpikeInterfaceSorting
    nwbfile: NWBFile
    property_descriptions: dict
        For each key in this dictionary which matches the name of a unit
        property in sorting, adds the value as a description to that
        custom unit column.
    skip_properties: list of str
        Each string in this list that matches a unit property will not be written to the NWBFile.
    skip_features: list of str
        Each string in this list that matches a spike feature will not be written to the NWBFile.
    write_in_processing_module: bool (optional, defaults to False)
        How to save the units table in the nwb file.
        - True will save it to the processing module to serve as a historical provenance for the official table.
        - False will save it to the official NWBFile.Units position; recommended only for the final form of the data.
    units_table_name : str (optional, defaults to 'units')
        The name of the units table. If write_as=='units', then units_table_name must also be 'units'.
    unit_table_description : str (optional)
        Text description of the units table; it is recommended to include information such as the sorting method,
        curation steps, etc.
    write_waveforms : bool (optional, defaults to false)
        if True and sorting is a spikeextractors SortingExtractor object then waveforms are added to the units table
        after writing.
    """
    if not isinstance(nwbfile, pynwb.NWBFile):
        raise TypeError(f"nwbfile type should be an instance of pynwb.NWBFile but got {type(nwbfile)}")

    if isinstance(sorting, SortingExtractor):
        checked_sorting = OldToNewSorting(oldapi_sorting_extractor=sorting)
    else:
        checked_sorting = sorting

    if write_in_processing_module:
        ecephys_mod = get_module(
            nwbfile=nwbfile,
            name="ecephys",
            description="Intermediate data from extracellular electrophysiology recordings, e.g., LFP.",
        )
        write_table_first_time = units_table_name not in ecephys_mod.data_interfaces
        if write_table_first_time:
            units_table = pynwb.misc.Units(name=units_table_name, description=unit_table_description)
            ecephys_mod.add(units_table)

        units_table = ecephys_mod[units_table_name]
    else:
        write_table_first_time = nwbfile.units is None
        if write_table_first_time:
            nwbfile.units = pynwb.misc.Units(name="units", description=unit_table_description)
        units_table = nwbfile.units

    default_descriptions = dict(
        isi_violation="Quality metric that measures the ISI violation ratio as a proxy for the purity of the unit.",
        firing_rate="Number of spikes per unit of time.",
        template="The extracellular average waveform.",
        max_channel="The recording channel id with the largest amplitude.",
        halfwidth="The full-width half maximum of the negative peak computed on the maximum channel.",
        peak_to_valley="The duration between the negative and the positive peaks computed on the maximum channel.",
        snr="The signal-to-noise ratio of the unit.",
        quality="Quality of the unit as defined by phy (good, mua, noise).",
        spike_amplitude="Average amplitude of peaks detected on the channel.",
        spike_rate="Average rate of peaks detected on the channel.",
        unit_name="Unique reference for each unit.",
    )
    if property_descriptions is None:
        property_descriptions = dict()
    if skip_properties is None:
        skip_properties = list()

    property_descriptions = dict(default_descriptions, **property_descriptions)

    data_to_add = defaultdict(dict)
    sorting_properties = checked_sorting.get_property_keys()
    excluded_properties = list(skip_properties) + ["contact_vector"] + ["sh"]
    properties_to_extract = [property for property in sorting_properties if property not in excluded_properties]

    # Extract properties
    for property in properties_to_extract:
        data = checked_sorting.get_property(property)
        index = isinstance(data[0], (list, np.ndarray, tuple))
        description = property_descriptions.get(property, "No description.")
        data_to_add[property].update(description=description, data=data, index=index)
        if property in ["max_channel", "max_electrode"] and nwbfile.electrodes is not None:
            data_to_add[property].update(table=nwbfile.electrodes)

    # Unit name logic
    units_ids = checked_sorting.get_unit_ids()
    if "unit_name" in data_to_add:
        unit_name_array = data_to_add["unit_name"]["data"]
    else:
        unit_name_array = units_ids.astype("str", copy=False)
        data_to_add["unit_name"].update(description="Unique reference for each unit.", data=unit_name_array)

    # If the channel ids are integer keep the old behavior of asigning table's id equal to unit_ids
    if np.issubdtype(units_ids.dtype, np.integer):
        data_to_add["id"].update(data=units_ids.astype("int"))

    units_table_previous_properties = set(units_table.colnames) - set({"spike_times"})
    extracted_properties = set(data_to_add)
    properties_to_add_by_rows = units_table_previous_properties | set({"id"})
    properties_to_add_by_columns = extracted_properties - properties_to_add_by_rows

    # Find default values for properties / columns already in the table
    type_to_default_value = {list: [], np.ndarray: np.array(np.nan), str: "", Real: np.nan}
    property_to_default_values = {"id": None}
    for property in units_table_previous_properties:
        # Find a matching data type and get the default value
        sample_data = units_table[property].data[0]
        matching_type = next(type for type in type_to_default_value if isinstance(sample_data, type))
        default_value = type_to_default_value[matching_type]
        property_to_default_values.update({property: default_value})

    # Add data by rows excluding the rows with previously added unit names
    unit_names_used_previously = []
    if "unit_name" in units_table_previous_properties:
        unit_names_used_previously = units_table["unit_name"].data

    properties_with_data = {property for property in properties_to_add_by_rows if "data" in data_to_add[property]}
    rows_in_data = [index for index in range(checked_sorting.get_num_units())]
    rows_to_add = [index for index in rows_in_data if unit_name_array[index] not in unit_names_used_previously]
    for row in rows_to_add:
        unit_kwargs = dict(property_to_default_values)
        for property in properties_with_data:
            unit_kwargs[property] = data_to_add[property]["data"][row]
        spike_times = checked_sorting.get_unit_spike_train(unit_id=units_ids[row], return_times=True)
        units_table.add_unit(spike_times=spike_times, **unit_kwargs, enforce_unique_id=True)

    # Add unit_name as a column and fill previously existing rows with unit_name equal to str(ids)
    previous_table_size = len(units_table.id[:]) - len(unit_name_array)
    if "unit_name" in properties_to_add_by_columns:
        cols_args = data_to_add["unit_name"]
        data = cols_args["data"]

        previous_ids = units_table.id[:previous_table_size]
        default_value = np.array(previous_ids).astype("str")

        extended_data = np.hstack([default_value, data])
        cols_args["data"] = extended_data
        units_table.add_column("unit_name", **cols_args)

    # Build  a channel name to electrode table index map
    table_df = units_table.to_dataframe().reset_index()
    unit_name_to_electrode_index = {
        unit_name: table_df.query(f"unit_name=='{unit_name}'").index[0] for unit_name in unit_name_array
    }

    indexes_for_new_data = [unit_name_to_electrode_index[unit_name] for unit_name in unit_name_array]
    indexes_for_default_values = table_df.index.difference(indexes_for_new_data).values

    # Add properties as columns
    for property in properties_to_add_by_columns - set({"unit_name"}):
        cols_args = data_to_add[property]
        data = cols_args["data"]
        if np.issubdtype(data.dtype, np.integer):
            data = data.astype("float")

        # Find first matching data-type
        sample_data = data[0]
        matching_type = next(type for type in type_to_default_value if isinstance(sample_data, type))
        default_value = type_to_default_value[matching_type]

        extended_data = np.empty(shape=len(units_table.id[:]), dtype=data.dtype)
        extended_data[indexes_for_new_data] = data

        extended_data[indexes_for_default_values] = default_value
        # Always store numpy objects as strings
        if np.issubdtype(extended_data.dtype, np.object_):
            extended_data = extended_data.astype("str", copy=False)
        cols_args["data"] = extended_data
        units_table.add_column(property, **cols_args)

    if write_waveforms:
        units_table = _add_waveforms_to_units_table(
            sorting=sorting, units_table=units_table, skip_features=skip_features
        )

def _add_waveforms_to_units_table(
    sorting: SortingExtractor,
    units_table,
    skip_features: Optional[List[str]] = None,
):
    """
    Auxiliar method for adding waveforms to an existing units_table.

    Parameters
    ----------
    sorting:  A spikeextractors SortingExtractor.
    units_table: a previously created units table
    skip_features: list of str
        Each string in this list that matches a spike feature will not be written to the NWBFile.
    """
    unit_ids = sorting.get_unit_ids()

    if isinstance(sorting, SortingExtractor):
        all_features = set()
        for unit_id in unit_ids:
            all_features.update(sorting.get_unit_spike_feature_names(unit_id))
        if skip_features is None:
            skip_features = []
        # Check that multidimensional features have the same shape across units
        feature_shapes = dict()
        for feature_name in all_features:
            shapes = []
            for unit_id in unit_ids:
                if feature_name in sorting.get_unit_spike_feature_names(unit_id=unit_id):
                    feat_value = sorting.get_unit_spike_features(unit_id=unit_id, feature_name=feature_name)
                    if isinstance(feat_value[0], (int, np.integer, float, str, bool)):
                        break
                    elif isinstance(feat_value[0], (list, np.ndarray)):  # multidimensional features
                        if np.array(feat_value).ndim > 1:
                            shapes.append(np.array(feat_value).shape)
                            feature_shapes[feature_name] = shapes
                    elif isinstance(feat_value[0], dict):
                        print(f"Skipping feature '{feature_name}' because dictionaries are not supported.")
                        skip_features.append(feature_name)
                        break
                else:
                    print(f"Skipping feature '{feature_name}' because not share across all units.")
                    skip_features.append(feature_name)
                    break
        nspikes = {k: get_nspikes(units_table, int(k)) for k in unit_ids}
        for feature_name in feature_shapes.keys():
            # skip first dimension (num_spikes) when comparing feature shape
            if not np.all([elem[1:] == feature_shapes[feature_name][0][1:] for elem in feature_shapes[feature_name]]):
                print(f"Skipping feature '{feature_name}' because it has variable size across units.")
                skip_features.append(feature_name)
        for feature_name in set(all_features) - set(skip_features):
            values = []
            if not feature_name.endswith("_idxs"):
                for unit_id in sorting.get_unit_ids():
                    feat_vals = sorting.get_unit_spike_features(unit_id=unit_id, feature_name=feature_name)
                    if len(feat_vals) < nspikes[unit_id]:
                        skip_features.append(feature_name)
                        print(f"Skipping feature '{feature_name}' because it is not defined for all spikes.")
                        break
                    else:
                        all_feat_vals = feat_vals
                    values.append(all_feat_vals)
                flatten_vals = [item for sublist in values for item in sublist]
                nspks_list = [sp for sp in nspikes.values()]
                spikes_index = np.cumsum(nspks_list).astype("int64")
                if feature_name in units_table:  # If property already exists, skip it
                    warnings.warn(f"Feature {feature_name} already present in units table, skipping it")
                    continue
                set_dynamic_table_property(
                    dynamic_table=units_table,
                    row_ids=[int(k) for k in unit_ids],
                    property_name=feature_name,
                    values=flatten_vals,
                    index=spikes_index,
                )
        else:
            """
            Currently (2022-04-22), spikeinterface does not support waveform extraction.
            """
            pass

    return units_table