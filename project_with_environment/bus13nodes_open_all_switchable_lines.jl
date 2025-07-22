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



nw_map = ieee13_data["network"]["nw"]

for t in sort(collect(keys(result_single)))   # 按升序遍历所有时间点
    println("=== 时刻 t = $t ===")
    # 1) 取出这一时刻的开关解
    switch_sol = result_single[t]["solution"]["switch"]
    # 2) 取出这一时刻对应的子网络元数据
    sub_net    = nw_map[t]

    for (sw_id, sol) in switch_sol
        state = sol["state"]                    # OPEN 或 CLOSED
        meta  = sub_net["switch"][sw_id]        # 该时刻子网里的 switch 元数据
        fbus  = meta["f_bus"]                   # 开关一端母线
        tbus  = meta["t_bus"]                   # 开关另一端母线

        @printf("  %s: %s ↔ %s is %s\n", sw_id, fbus, tbus, state)
    end
    println()  # 空行分隔不同时间点
end

using Graphs, GraphIO  
res = result_single  # 已经跑完 optimize_switches，并且包含所有 t 的结果
# 2) 收集所有时刻解里出现的母线列表
all_buses = unique(vcat(
    [ collect(keys(res[t]["solution"]["bus"])) for t in keys(res) ]...
))
# 建 ID→索引 的映射
bus_to_idx = Dict(bus => i for (i, bus) in enumerate(all_buses))

# 3) 预先把“永闭合”的线路加入模板图 g0
base_net = ieee13_data["network"]
t0 = sort(collect(keys(res)))[1]
g0 = SimpleGraph(length(all_buses))
for (_, ln) in ieee13_data["network"]["nw"][t0]["line"]
    # ln["f_bus"]、ln["t_bus"] 是该条线路两端的母线 ID
    u = bus_to_idx[ln["f_bus"]]
    v = bus_to_idx[ln["t_bus"]]
    add_edge!(g0, u, v)
end

# 4) 对每个时间点 t，克隆 g0，加上当时闭合的开关，并写 GraphML
for t in sort(collect(keys(res)))
    gt = deepcopy(g0)

    # 4.1 当前时刻的网络元数据
    sub_net = ieee13_data["network"]["nw"][t]

    # 4.2 把闭合的开关边加入
    for (sw, sol) in res[t]["solution"]["switch"]
        state = sol["state"]
        # 有时 state 是 Symbol :CLOSED，有时是 Bool true
        if state == :CLOSED || state == "CLOSED" || state === true
            meta = sub_net["switch"][sw]
            fbus = meta["f_bus"]
            tbus = meta["t_bus"]
            add_edge!(gt, bus_to_idx[fbus], bus_to_idx[tbus])
        end
    end

    # 4.3 输出
    outfile = "ieee13_t$(t).graphml"
    savegraph(outfile, gt)
    println("已生成：", outfile)
end

eng = ONM.PMD.parse_file(dss_file)# 举例取第 1 个时刻的子网络
# 构造一个 EzXML.Document 对象
doc = PowerModelsONM.build_graphml_document(eng; type="unnested")
# 把它写到文件
open("ieee13_nested.graphml", "w") do io
    print(io, doc)
end
println("生成成功：ieee13_t1_pm.graphml")

PowerModelsONM.save_graphml(
  "ieee13_t1_unnested.graphml",
  ieee13_data["network"]["nw"]["1"];
  type="unnested"
)
println("已生成简化版：ieee13_t1_unnested.graphml")

# TODO 生成result_single结果的图

# TODO 改变solar和storage，看看optimal configuration会不会变化