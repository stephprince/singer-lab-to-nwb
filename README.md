# singer-lab-to-nwb

## overview

This repository contains code for converting singer lab data to nwb format

## how to use

To install packages in development mode

```python
cd /path/to/files/singer-lab-to-nwb/
pip install -e singer_lab_to_nwb
```

The main conversion script is `convert_singer_lab_data.py`. This script uses the nwb_conversion_tools workflow.
In this workflow, DataInterface classes are built to extract data from the multiple data formats.
The NWBConverter class is then used to combine the  multiple data formats into the final NWB file.

This repository is under development and is currently being tested on individual files. Eventually the format will be
updated to process multiple files and pull this session info from summary spreadsheets.

## how to adapt

For other Singer Lab users, the SingerLabNWBConverter could be used as the base converter, and individuals add different
data interfaces to combine other data types as needed (e.g., Intan files instead of SpikeGadgets or other virmen tasks)