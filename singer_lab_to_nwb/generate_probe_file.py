import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from probeinterface import Probe, ProbeGroup
from probeinterface.io import write_probeinterface
from probeinterface.plotting import plot_probe_group, plot_probe

# import channel coordinates and mappingfrom csv file
coord_file_path="Y:\singer\Steph\Code\singer-lab-to-nwb\data\ProbeData\A2x32-Poly5-10mm-20s-200-100-probemap.csv"
df_coord = pd.read_csv(coord_file_path)
wiring_file_path = "Y:\singer\Steph\Code\singer-lab-to-nwb\data\ProbeData\SpikeGadgets_A2x32Poly5_Mapping_210327_RigC.csv"
df_wiring = pd.read_csv(wiring_file_path)

# generate probe objects
probe1 = Probe(ndim=2, si_units='um')
probe1.set_contacts(positions=df_coord[["X","Y"]], shapes='circle', shape_params={'radius': np.sqrt(100/np.pi)}, shank_ids=df_coord[["K"][0]])
probe2 = probe1.copy()

# create probe group (2 probes, 1 in each brain region)
probegroup = ProbeGroup()
probegroup.add_probe(probe1)
probegroup.add_probe(probe2)

# add device wiring info
probe1.set_device_channel_indices(df_wiring["HW Channel"][0:64].values)
probe2.set_device_channel_indices(df_wiring["HW Channel"][64:128].values)

# visualize probe object
plot_probe_group(probegroup, with_channel_index=True, same_axes=False)

fig, ax = plt.subplots()
plot_probe(probe1, ax=ax, with_channel_index=True, with_device_index=True)
plt.show()

# save the probe object
write_probeinterface("../data/ProbeData/update-task-64-chan-dual-probes.json", probegroup)
