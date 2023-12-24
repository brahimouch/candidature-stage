
from __future__ import annotations

import numpy as np
import pandas as pd
from data.utils import average_timeseries
from data.utils import get_pvgis_tmy_from_location
from data.utils import load_example_data
from disciplines import CostDiscipline
from disciplines import PyMGridDiscipline
from disciplines import SimpleWTDiscipline
from disciplines import SolarPanelDiscipline
from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from OptiUP.LCAProblem import LCADiscipline



configure_logger()

# Fixed values
STUDY_TIME = 1  # in years
# Lifetime of components
LT_BAT_LIION = 3
LT_PV = 30
LT_WT = 25

# Price of components
PRICE_PV = 150.0 / LT_PV  # eur/m2/year
PRICE_WT = 1500.0 / LT_WT  # eur/unit/year
PRICE_BAT = 100.0 / LT_BAT_LIION  # eur/kWh/year
PRICE_ELEC = 0.01  # eur/kWh/year

### Prepare unmutable data to be used in the analysis
days_to_consider = [240, 330, 3]
weights_of_days = [180.0, 135.0, 50.0]
analysis_period = 24 * len(days_to_consider)  # hours

# Let's create "analysis_period" days worth of hourly data
# wind_ts = 20 * np.random.rand(analysis_period)  # random pv data in the range [0, 25]
consumptions, irradiance = load_example_data(indexes=days_to_consider)
load_ts = np.concatenate(consumptions, axis=None)
# grenoble_data = pd.read_csv(
#     "data/irradiance_and_consumption_data.csv", sep=",", decimal="."
# )
# load_ts = grenoble_data["Consumption"].to_numpy()
# grenoble_data.TIMESTAMP = grenoble_data.TIMESTAMP.apply(pd.to_datetime, "%Y-%m-%d %H")
# grenoble_data.set_index(grenoble_data.TIMESTAMP, inplace=True, drop=True)
# solar_ts = [pd.DataFrame(
#         {"DNI": grenoble_data["DNI"], "DHI": grenoble_data["DHI"]}, index=grenoble_data.index
#     ).fillna(0)]

# # Get weather data for Toulouse
coordinates = (43.60, 1.44)  # degrees N, degrees E Toulouse
columns = ["dni", "dhi", "wind_speed"]
df = get_pvgis_tmy_from_location(*coordinates)  # already hourly data
wind_ts = np.concatenate(
    df.groupby([df.index.day_of_year, df.index.hour])["wind_speed"]
    .mean()
    .unstack()
    .values[days_to_consider, :],
    axis=None,
)


# %%

### Setup design variables, objectives, and constraints
# variables: list of dicts, var = {"name": "var1", "l_b": -10., "u_b": 10., "value": np.array([0.])}

# Definition of units.
# pv_surface - square meters
# pv_tilt - degrees
# pv_azimuth - degrees
# bat_max_capacity - kilowatt-hour

variables = [
    {"name": "pv_surface", "l_b": 1.0, "u_b": 80.0, "value": np.array([10.0])},
    {"name": "pv_tilt", "l_b": 0.0, "u_b": 90.0, "value": np.array([45.0])},
    {"name": "pv_azimuth", "l_b": -80, "u_b": 80.0, "value": np.array([0.0])},
    {"name": "n_wt", "l_b": 0, "u_b": 3, "value": np.array([0]), "type": "integer"},
    # {"name": "n_houses", "l_b": 1, "u_b": 15, "value": np.array([1]), "type": "integer"},
    {
        "name": "bat_max_capacity",
        "l_b": 10.0,
        "u_b": 100.0,
        "value": np.array([10.0]),
    },
]
# objectives: list of strings, "obj_name"
objectives = ["grid-import"]

# constraints: list of dicts, c = {"name": "c1", "type": "eq" or "ineq"}; expressions not handled yet
constraints = []

### Create disciplines
disciplines = []

# Discipline 1: Solar Panel Production
# This discipline outputs results in Watts
# For a timestep of 1hour, it is equivalent to Watt-hour

solar_disc = SolarPanelDiscipline(
    panel_data={"pv_tilt": 10, "pv_azimuth": 0, "pv_surface": 1},
    solar_data=irradiance,
    local_data={"latitude": 45.2, "longitude": -5.7},
    output_name="pv_time_series",
)

solar_disc.input_grammar.update(["pv_surface", "pv_tilt", "pv_azimuth"])
solar_disc.output_grammar.update(["pv_time_series"])
disciplines.append(solar_disc)

# %%

# Discipline 2: Wind Energy Production
wt_data = {"v_cutin": 3, "v_cutout": 25, "v_rat": 11, "p_rat": 5000}
wind_disc = SimpleWTDiscipline(
    wt_data, wind_ts, ts_factor=1.0, output_name="wt_time_series"
)
solar_disc.input_grammar.update(["n_wt"])
wind_disc.output_grammar.update(["wt_time_series"])
disciplines.append(wind_disc)


# Discipline 3: Load discipline
# This discipline is used if we want to let the optimizer select the number of
# houses the grid provides energy to
def f_load(n_houses=1):
    load_time_series = load_ts * n_houses
    return load_time_series


load_disc = create_discipline("AutoPyDiscipline", py_func=f_load)
# disciplines.append(load_disc)

# %%

# Discipline 3: MicroGrid
# We also need a grid module. It requires, as input, a time series with three to
# four columns: [import price, export price, CO2 prod per KWh] + optional [grid status: bool], defaults to True
# The time-series should be given as hourly kWh values, hence the scale factor in the pv, wt, and load modules
grid_ts = [0.2, 0.1, 0.5] * np.ones((analysis_period, 3))
fixed_modules = [
    {
        "name": "load",
        "function": "LoadModule",
        "options": {"time_series": load_ts * 3, "scale": 0.001},
    },
    {
        "name": "grid",
        "function": "GridModule",
        "options": {"time_series": grid_ts, "max_import": 1000, "max_export": 1000},
    },
    # {"name": , "function": , "options": {}},
]
opt_modules = [
    {"name": "pv", "function": "RenewableModule", "options": {"scale": 0.001}},
    {"name": "wt", "function": "RenewableModule", "options": {"scale": 0.001}},
    # {
    #     "name": "load",
    #     "function": "LoadModule",
    #     "options": {"scale": 0.001},
    # },
    {
        "name": "bat",
        "function": "BatteryModule",
        "options": dict(
            min_capacity=1.0,
            # OPTIMIZED:max_capacity=100,
            max_charge=50,
            max_discharge=50,
            efficiency=0.9,
            init_soc=0.4,
        ),
    },
]
microgrid_options = {"timeseries_length": analysis_period}
run_options = {"verbose": False}
mgrid_disc = PyMGridDiscipline(
    fixed_modules,
    opt_modules,
    microgrid_options,
    run_options,
    split_by=24,
    split_weights=weights_of_days,
)
mgrid_disc.input_grammar.update(
    ["bat_max_capacity", "pv_time_series", "wt_time_series"]
)
mgrid_disc.output_grammar.update(["grid-import"])
disciplines.append(mgrid_disc)

# %%
# Discipline 4: LCA / Environmental Impact
LCA_PROJ_NAME = "lco"
LCA_CS_NAME = "calcul_microgrid"

# Define activities and methods
LCA_CS_ACTIVITY_LIST = [
    {
        "name": "Microgrid with 5kw and 100kw wind turbines, solar panel, battery and grid",
        "loc": "FRA",
        "db_name": "Microgrid",
    },
]
LCA_CS_METHODS_LIST = [
    ("ReCiPe Midpoint (H)", "climate change", "GWP100"),
    ("ReCiPe Midpoint (H)", "water depletion", "WDP"),
    ("ReCiPe Midpoint (H)", "urban land occupation", "ULOP"),
]

# Create calculation setup
if True:
    import brightway2 as bw
    import lca_algebraic as lcaa

    bw.projects.set_current(LCA_PROJ_NAME)
    bw.calculation_setups[LCA_CS_NAME] = {
        "inv": [{lcaa.findActivity(**a): 1} for a in LCA_CS_ACTIVITY_LIST],
        "ia": [lcaa.findMethods(search=str(m))[0] for m in LCA_CS_METHODS_LIST],
    }

LCA_disc = LCADiscipline(
    cs_name=LCA_CS_NAME,
    proj_name=LCA_PROJ_NAME,
    name="LCA",
    compute_mode="algebraic",  # this is here to make use of lca_algebraic
    keymap={
        "bat_max_capacity": "bat_capacity",
        "n_wt": "n_5kw_wt",
        "grid-import": "grid_import",
    },  # {gemseo_variable: lca_variable}
)
LCA_methods = LCA_disc.get_method_names()
LCA_disc.input_grammar.update(["pv_surface", "bat_max_capacity", "grid-import", "n_wt"])
LCA_disc.output_grammar.update(LCA_methods)
disciplines.append(LCA_disc)
objectives += LCA_methods

# %%
# Discipline 4: Economical Cost
# prices are multiplied by inputs with same keyword and added together
cost_disc = CostDiscipline(
    prices={
        "bat_max_capacity": PRICE_BAT,
        "pv_surface": PRICE_PV,
        "n_wt": PRICE_WT,
        "grid-import": PRICE_ELEC,
    }
)

cost_disc.input_grammar.update(
    ["pv_surface", "bat_max_capacity", "grid-import", "n_wt"]
)
cost_disc.output_grammar.update(["cost"])
disciplines.append(cost_disc)
objectives += ["cost"]

### Create design space
design_space = create_design_space()
for var in variables:
    design_space.add_variable(
        var["name"],
        l_b=var["l_b"],
        u_b=var["u_b"],
        value=var.get("value", np.ones(1)),
        var_type=var.get("type", "float"),
    )
