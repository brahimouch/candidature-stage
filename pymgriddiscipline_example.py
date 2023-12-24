
# %%
import numpy as np
import pandas as pd

np.random.seed(0)

from pymgrid import Microgrid
from pymgrid.modules import BatteryModule, LoadModule, RenewableModule, GridModule
from pymgrid.algos import RuleBasedControl

from gemseo.core.discipline import MDODiscipline

from disciplines import PyMGridDiscipline


# %%
analysis_period = 365  # in days

# Let's create 90 days worth of hourly data
load_ts = 100 + 100 * np.random.rand(
    24 * analysis_period
)  # random load data int he range [100, 200]
pv_ts = 200 * np.random.rand(
    24 * analysis_period
)  # random pv data in the range [0, 200]

# We also need a grid module. It requires, as input, a time series with three to
# four columns: [import price, export price, CO2 prod per KWh] + optional [grid status: bool], defaults to True
grid_ts = [0.2, 0.1, 0.5] * np.ones((24 * analysis_period, 3))

fixed_modules = [
    {"name": "pv", "function": "RenewableModule", "options": {"time_series": pv_ts}},
    {"name": "load", "function": "LoadModule", "options": {"time_series": load_ts}},
    {
        "name": "grid",
        "function": "GridModule",
        "options": {"time_series": grid_ts, "max_import": 100, "max_export": 100},
    },
    {
        "name": "bat_1",
        "function": "BatteryModule",
        "options": dict(
            min_capacity=10,
            max_capacity=100,
            max_charge=50,
            max_discharge=50,
            efficiency=0.9,
            init_soc=0.2,
        ),
    },
    {
        "name": "bat_2",
        "function": "BatteryModule",
        "options": dict(
            min_capacity=10,
            max_capacity=1000,
            max_charge=10,
            max_discharge=10,
            efficiency=0.7,
            init_soc=0.2,
        ),
    },
    # {"name": , "function": , "options": {}},
]
opt_modules = {}
microgrid_options = {"timeseries_length": 24 * analysis_period}
run_options = {}

mgrid = PyMGridDiscipline(
    fixed_modules, opt_modules, microgrid_options, run_options, split_by=None
)

mgrid.output_grammar.update(["grid_import", "grid_export"])

mgrid._run()

# %%
