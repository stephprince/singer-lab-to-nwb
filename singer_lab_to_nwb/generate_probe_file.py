import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from probeinterface import Probe, ProbeGroup
from probeinterface.io import write_prb
from probeinterface.plotting import plot_probe_group

# import channel mapping from csv file
file_path="Y:\singer\Steph\Code\singer-lab-to-nwb\data\ProbeData\A2x32-Poly5-10mm-20s-200-100-probemap.csv"
df = pd.read_csv(file_path)

# generate probe objects
probe1 = Probe(ndim=2, si_units='um')
probe1.set_contacts(positions=df[["X","Y"]], shapes='circle', shape_params={'radius': np.sqrt(100/np.pi)}, shank_ids=df[["K"][0]])
probe2 = probe1.copy()

# create probe group (2 probes, 1 in each brain region)
probegroup = ProbeGroup()
probegroup.add_probe(probe1)
probegroup.add_probe(probe2)

probe1.set_device_channel_indices(np.arange(0,64))
probe2.set_device_channel_indices(np.arange(64,128))

# visualize probe object
plot_probe_group(probegroup, with_channel_index=True, same_axes=False)
plt.show()

# save the probe object
write_prb("../data/ProbeDataupdate-task-64-chan-dual-probes.prb", probegroup)
