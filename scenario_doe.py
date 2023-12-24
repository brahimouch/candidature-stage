
# %%
from gemseo.api import create_scenario
from problem_definition import constraints
from problem_definition import design_space
from problem_definition import disciplines
from problem_definition import objectives
from problem_definition import variables

### Create scenario
scenario = create_scenario(
    disciplines,
    formulation="DisciplinaryOpt",
    objective_name=objectives[0],
    design_space=design_space,
    scenario_type="DOE",
)
# Scenario options
scenario_options = {"algo": "lhs", "n_samples": 10}
# Add the remainaing objectives as constraints (required for Pareto front visualization)
for o in objectives[1:]:
    scenario.add_constraint(o, "ineq")

### Create constraints
for c in constraints:
    scenario.add_constraint(c["name"], c["type"])

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
    scenario.post_process("ParetoFront", objectives=objectives, save=True, show=True)

    scenario.save_optimization_history("my_results.hdf")
# %%
