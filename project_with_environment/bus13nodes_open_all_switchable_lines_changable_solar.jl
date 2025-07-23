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
settings_file = joinpath(onm_path, "ieee13_settings_open_all_switchable_lines.json")
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

# 定义一组光伏出力 P 和储能能量 E 的场景
solar_levels   = 200   # kW  在 IEEE 13 节点等典型配电网案例里，单节点 PV 容量一般选在 10 kW – 50 kW
storage_levels = 100   # kWh

results = Dict{Tuple{Float64,Float64}, Any}()

for P_solar in solar_levels, E_stor in storage_levels
    net = deepcopy(ieee13_data["network"])

    for t in keys(net["nw"])
        
        nt = length(keys(net["nw"])) 

        # —— 修改光伏出力上限 —— #
        # dss 字段里包含所有逆变器资产：PVSystem.PV_mg1a, PV_mg1b…… :contentReference[oaicite:4]{index=4}
        for (asset, inv) in net["nw"][t]["solar"]
            if startswith(asset, "pv_mg")
                # 假设每个时刻都按同样上限
                println("original pg:", inv["pg_ub"])
                inv["pg_lb"] = fill(P_solar, nt)  # 下界=上界 目标是不允许对光伏做任何形式的切除（curtailment）这样模型里就默认“能发多少就发多少”，而不是留空让优化认为“不用发也行”。
                inv["pg_ub"] = fill(P_solar, nt)  # 上界
                println("after: ", inv["pg_ub"])
            end
        end

        # # —— 修改储能初始能量或能量上限 —— #
        # # storage 字段里有 battery_mg1a, b, c 等 :contentReference[oaicite:5]{index=5}
        # for (batt, bcfg) in net["nw"][t]["storage"]
        #     bcfg["energy_lb"] = 0.0
        #     # bcfg["energy"]    = E_stor     # 初始能量
        #     bcfg["energy_ub"] = E_stor   # 能量容量上限
        #     # 若需限速可同理设置 charge_rating/discharge_rating
        # end

    end

    # 运行一次最优开关 / MLD 问题
    sol = ONM.optimize_switches(
        net,
        ieee13_mip_solver;
        formulation = ONM.LPUBFDiagPowerModel,
        algorithm   = "rolling-horizon"  # 或 "full-lookahead"
    )  # 默认 problem="block" 即 MLD-Block :contentReference[oaicite:2]{index=2}

    # 存储每个场景下的开关配置（比如 sol["switching_actions"]）
    # results[(P_solar, E_stor)] = sol["1"]["solution"]["switch"]
end

using JSON

open("open_all_lines_new_solar_200_lbubeqal.json", "w") do io
    JSON.print(io, sol)
end

############ optimal configuration output #############

nw_map = ieee13_data["network"]["nw"]

using Printf

for t in sort(collect(keys(sol)))   # 按升序遍历所有时间点
    println("=== 时刻 t = $t ===")
    # 1) 取出这一时刻的开关解
    switch_sol = sol[t]["solution"]["switch"]
    # 2) 取出这一时刻对应的子网络元数据
    sub_net    = nw_map[t]

    for (sw_id, solution) in switch_sol
        state = solution["state"]                    # OPEN 或 CLOSED
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

# TODO 改变solar和storage，看看optimal configuration会不会变化

# TODO 生成result_single结果的图

