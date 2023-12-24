
# This script intents to serve as an example of how to use the python-microgrid
# library. Further information is found in the documentation.
# %%
import matplotlib.pyplot as plt
import numpy as np
from pymgrid import Microgrid
from pymgrid.algos import RuleBasedControl
from pymgrid.modules import BatteryModule
from pymgrid.modules import GridModule
from pymgrid.modules import LoadModule
from pymgrid.modules import RenewableModule


np.random.seed(0)

# Define some input parameters

analysis_period = 1  # in days


# Lets create two battery modules, one small and other large, with different
# charging rates
small_battery = BatteryModule(
    min_capacity=10,
    max_capacity=100,
    max_charge=50,
    max_discharge=50,
    efficiency=0.9,
    init_soc=0.2,
)

large_battery = BatteryModule(
    min_capacity=10,
    max_capacity=1000,
    max_charge=10,
    max_discharge=10,
    efficiency=0.7,
    init_soc=0.2,
)

# Let's create 90 days worth of hourly data
load_ts = 100 + 100 * np.random.rand(
    24 * analysis_period
)  # random load data int he range [100, 200]
pv_ts = 200 * np.random.rand(
    24 * analysis_period
)  # random pv data in the range [0, 200]

# And then create the load and photovoltaic modules
load = LoadModule(time_series=load_ts)
pv = RenewableModule(time_series=np.array(pv_ts))

# We also need a grid module. It requires, as input, a time series with three to
# four columns: [import price, export price, CO2 prod per KWh] + optional [grid status: bool], defaults to True
grid_ts = [0.2, 0.1, 0.5] * np.ones((24 * analysis_period, 3))
grid = GridModule(max_import=100, max_export=100, time_series=grid_ts)

microgrid = Microgrid(modules=[small_battery, large_battery, pv, load, grid])

rbc = RuleBasedControl(microgrid)

rbc.reset()
rbc_results = rbc.run()

# %%

rbc_results[
    [
        ("load", 0, "load_current"),
        ("renewable", 0, "renewable_used"),
        ("grid", 0, "grid_import"),
    ]
].plot()
plt.show()
