import pandas as pd

from pathlib import Path

def get_file_paths(base_path, session_id):
    """
    sets default file paths for different data types
    """

    raw_ephys_path = base_path / "RawData" / "UpdateTask" / session_id
    processed_ephys_path = base_path / "ProcessedData" / "UpdateTask" / session_id
    virmen_path = base_path / "Virmen Logs" / "UpdateTask"
    kilosort_path = Path("sorted/kilosort/new_unit_ids")
    cell_explorer_path = processed_ephys_path / 'cellTypeClassification.mat'
    channel_map_path = Path("Y:/singer/Steph/Code/singer-lab-to-nwb/data/ProbeData/A2x32-Poly5-10mm-20s-200-100-RigC-ProbeMap.csv")
    local_path = Path("C:/Users/sprince7/Documents")
    nwbfile_path = str(local_path / "DANDIdata" / "001371" / f"{session_id}.nwb")

    return dict(raw_ephys=raw_ephys_path,
                processed_ephys=processed_ephys_path,
                virmen=virmen_path,
                kilosort=kilosort_path,
                cell_explorer=cell_explorer_path,
                channel_map=channel_map_path,
                session_id=session_id,
                nwbfile=nwbfile_path
                )


def get_session_info(filename, animals, dates_included=None, dates_excluded=None, behavior=None):
    # import all session info
    df_all = pd.read_csv(filename, skiprows=[1])  # skip the first row that's just the detailed header info

    # if None values, deal with appropriately so it doesn't negatively affect the filtering
    dates_incl = dates_included or df_all['Date']                   # if no value given, include all dates
    dates_excl = dates_excluded or [None]                           # if no value given, exclude no dates
    behavior = behavior or df_all['Behavior'].unique()              # if no value given, include all behavior types

    # filter session info depending on cases
    session_info = df_all[(df_all['Include'] == 1) &                # DOES HAVE an include value in the column
                          (df_all['Animal'].isin(animals)) &        # IS IN the animals list
                          (df_all['Date'].isin(dates_incl)) &       # IS IN the included dates list
                          ~(df_all['Date'].isin(dates_excl)) &  # NOT IN the excluded dates list
                          (df_all['Behavior'].isin(behavior))       # IS IN the behavior type list
                          ]

    return session_info


def get_module(nwbfile, name, description=None):
    """
    Check if processing module exists. If not, create it. Then return module.
    Parameters
    ----------
    nwbfile: pynwb.NWBFile
    name: str
    description: str | None (optional)
    Returns
    -------
    pynwb.module
    """

    if name in nwbfile.processing:
        return nwbfile.processing[name]
    else:
        if description is None:
            description = name
        return nwbfile.create_processing_module(name, description)


def get_spikegadgets_electrode_metadata(brain_regions, channel_groups):
    spikegadgets = [dict(
        name="spikegadgets_mcu",
        description="Two NeuroNexus silicon probes with 2 (shanks) x 32 (channels) on each probe were inserted into"
                    " hippocampal CA1 and medial prefrontal cortex. Probes were 64-chan, poly5 Takahashi probe "
                    "formats. Electrophysiological data were acquired using a SpikeGadgets MCU system digitized "
                    "with 30 kHz rate. Analog and digital channels were acquired using the SpikeGadgets ECU system."
    ),
        dict(name="spikegadgets_ecu",
             description="Analog and digital inputs of SpikeGadgets system. Max -10 to 10V for analog channels.")
    ]

    electrode_group = [dict(
        name=f'probe{n}',
        description=f'probe{n} of NeuroNexus probes.  Channels 0-31 belong to shank 1 and channels 32-64 '
                    f'belong to shank 2',
        location=brain_regions[n],
        device='spikegadgets_mcu')
        for n, _ in enumerate(channel_groups)
    ]
    electrode_group.append(dict(
        name='analog_inputs',
        description='analog inputs to SpikeGadgets system. Channel IDs are unique to the project and task.',
        location='none',
        device='spikegadgets_ecu')
    )

    sg_electrode_dict = dict(Device=spikegadgets,
                             ElectrodeGroup=electrode_group,)

    return sg_electrode_dict