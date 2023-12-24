
# %%
from gemseo.api import create_scenario
from problem_definition import constraints
from problem_definition import design_space
from problem_definition import disciplines
from problem_definition import LCA_methods
from problem_definition import objectives
from problem_definition import variables

### Create scenario
scenario = create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    objective_name=objectives,
    design_space=design_space,
)

### Create constraints
for c in constraints:
    scenario.add_constraint(c["name"], c["type"])

# Scenario options
tolerances = {
    "ftol_rel": 0.0,
    "fotl_abs": 0.0,
    "xtol_rel": 0.0,
    "xtol_abs": 0.0,
}
options = {
    **tolerances,
    "pop_size": 25,
    "stop_crit_n_x": 60,
    "stop_crit_n_hv": 5,
    "max_gen": 100,
}
scenario_options = {
    "algo": "PYMOO_NSGA2",
    "max_iter": options["pop_size"] * options["max_gen"] + 1,
    "algo_options": options,
}

if __name__ == "__main__":
    ### Execute scenario
    scenario.execute(scenario_options)

    ### Post-process scenario
    scenario.post_process(
        "ScatterPlotMatrix",
        variable_names=[v["name"] for v in variables],
        save=False,
        show=True,
    )

    labels = (
        ["Grid Imp."] + [method.split(",")[-1] for method in LCA_methods] + ["Cost"]
    )

    scenario.post_process("ParetoFront", objectives_labels=labels, save=True, show=True)

    scenario.save_optimization_history(
        f"opt_popsize={options['pop_size']}_maxgen={options['max_gen']}.hdf"
    )
# %%
