# https://lanl-ansi.github.io/PowerModelsONM.jl/stable/tutorials/Use%20Case%20Examples.html
# Use case example for ieee13 

begin 
    import DataFrames as DF
    import CSV 
    import PowerModelsONM as ONM
end
using PowerModelsONM, JuMP

# 1) Point to the three files on disk:
onm_path = "G:\\My Drive\\Research\\Bus_git\\Bus_system_case_study_reconfiguration"
dss_file      = joinpath(onm_path, "network.ieee13mod.dss")
settings_file = joinpath(onm_path, "ieee13_settings open_all_switchable_lines.json")
events_file   = joinpath(onm_path, "ieee13_events_open_all_switchable_lines.json")

# 2) Build the multinetwork data structure.
#    Since events.json is just [], there will be no forced changes mid‐run.
ieee13_data = ONM.prepare_data!(
    Dict{String,Any}(
      "network"  => dss_file,
      "settings" => settings_file,
      "events"   => events_file
    )
)

# 3) Extract the "network" sub‐dictionary (deep copy, so we don't modify the original in memory).
ieee13_mn = deepcopy(ieee13_data["network"])

# 4) Build solver handles. Here we grab the MIP solver from settings.
ieee13_mip_solver = ONM.build_solver_instances(
    solver_options = ieee13_data["settings"]["solvers"]
)["mip_solver"]

# 5) Run ONE‐STEP optimization.
#    Because time_elapsed = [1.0], ONM only solves for t = 1 and never loops.
# https://lanl-ansi.github.io/PowerModelsONM.jl/stable/reference/prob.html#Optimal-Switching-/-Maximal-Load-Delivery-(MLD)
result_single = ONM.optimize_switches(
    ieee13_mn,
    ieee13_mip_solver;
    formulation = ONM.LPUBFDiagPowerModel,
    algorithm   = "rolling-horizon",  # with a single timestep, this simply does one solve
    problem     = "block"             # standard branch‐and‐bound for mixed‐integer
)

using JSON

open("all_switchable_lines_results.json", "w") do io
    JSON.print(io, result_single)
end

