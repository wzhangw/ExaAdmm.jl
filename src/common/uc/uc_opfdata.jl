using Random

mutable struct UCGener
  # .gen fields
  bus::Int
  Pg::Float64
  Qg::Float64
  Qmax::Float64
  Qmin::Float64
  Vg::Float64
  mBase::Float64
  status::Int
  Pmax::Float64
  Pmin::Float64
  Pc1::Float64
  Pc2::Float64
  Qc1min::Float64
  Qc1max::Float64
  Qc2min::Float64
  Qc2max::Float64
  ramp_agc::Float64
  ru::Float64
  rd::Float64
  minUp::Int
  minDown::Int
  # .gencost fields
  gentype::Int
  startup::Float64
  shutdown::Float64
  n::Int
  coeff::Array
end

struct UCOPFData
  buses::Array{Bus}
  lines::Array{Line}
  load::Load{Matrix{Float64}}
  generators::Array{UCGener}
  bus_ref::Int
  baseMVA::Float64
  BusIdx::Dict{Int,Int}    #map from bus ID to bus index
  FromLines::Array         #From lines for each bus (Array of Array)
  ToLines::Array           #To lines for each bus (Array of Array)
  BusGenerators::Array     #list of generators for each bus (Array of Array)
end

function uc_initialize(gen::UCGener)::Nothing
  typenum = rand(1:3)
  if typenum == 1
    gen.ru        = max(gen.Pmin, gen.Pmax / 2)
    gen.rd        = max(gen.Pmin, gen.Pmax / 2)
    gen.minUp     = 2
    gen.minDown   = 2
  elseif typenum == 2
    gen.ru        = max(gen.Pmin, gen.Pmax / 3)
    gen.rd        = max(gen.Pmin, gen.Pmax / 3)
    gen.minUp     = 3
    gen.minDown   = 3
  else
    gen.ru        = max(gen.Pmin, gen.Pmax / 5)
    gen.rd        = max(gen.Pmin, gen.Pmax / 5)
    gen.minUp     = 4
    gen.minDown   = 4
  end
  return
end

function uc_opf_loaddata_matpower(case_name, load_prefix, lineOff=Line(); VI=Array{Int}, VD=Array{Float64}, case_format="matpower")
  Random.seed!(35)
  data = parse_matpower(case_name; case_format=case_format)

  #
  # Load buses
  #

  nbus = length(data["bus"])
  buses = Array{Bus}(undef, nbus)
  bus_ref = -1

  for i=1:nbus
    @assert data["bus"][i]["bus_i"] > 0
    buses[i] = Bus(data["bus"][i]["bus_i"],
                   data["bus"][i]["type"],
                   data["bus"][i]["Pd"],
                   data["bus"][i]["Qd"],
                   data["bus"][i]["Gs"],
                   data["bus"][i]["Bs"],
                   data["bus"][i]["area"],
                   data["bus"][i]["Vm"],
                   data["bus"][i]["Va"],
                   data["bus"][i]["baseKV"],
                   data["bus"][i]["zone"],
                   data["bus"][i]["Vmax"],
                   data["bus"][i]["Vmin"])
      if buses[i].bustype == 3
        if bus_ref > 0
          error("More than one reference bus present in the data")
        else
          bus_ref = i
        end
      end
  end

  #
  # Load branches
  #
  nline = length(data["branch"])
  lines = Array{Line}(undef, nline)
  for i=1:nline
    @assert data["branch"][i]["status"] == 1
    lines[i] = Line(data["branch"][i]["fbus"],
                    data["branch"][i]["tbus"],
                    data["branch"][i]["r"],
                    data["branch"][i]["x"],
                    data["branch"][i]["b"],
                    data["branch"][i]["rateA"],
                    data["branch"][i]["rateB"],
                    data["branch"][i]["rateC"],
                    data["branch"][i]["ratio"],
                    data["branch"][i]["angle"],
                    data["branch"][i]["status"],
                    data["branch"][i]["angmin"],
                    data["branch"][i]["angmax"])
  end

  #
  # Load generators
  #
  ngen = length(data["gen"])
  generators = Array{UCGener}(undef, ngen)
  for i=1:ngen
    generators[i] = UCGener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, Array{Int}(undef, 0)) #gen_arr[i,1:end]...)
    generators[i].bus = data["gen"][i]["bus"]
    generators[i].Pg = data["gen"][i]["Pg"]
    generators[i].Qg = data["gen"][i]["Qg"]
    generators[i].Qmax = isinf(data["gen"][i]["Qmax"]) ? 999.99 : data["gen"][i]["Qmax"]
    generators[i].Qmin = isinf(data["gen"][i]["Qmin"]) ? -999.99 : data["gen"][i]["Qmin"]
    generators[i].Vg = data["gen"][i]["Vg"]
    generators[i].mBase = data["gen"][i]["mBase"]
    generators[i].status = data["gen"][i]["status"]
    @assert generators[i].status == 1
    generators[i].Pmax = isinf(data["gen"][i]["Pmax"]) ? 999.99 : data["gen"][i]["Pmax"]
    generators[i].Pmin = isinf(data["gen"][i]["Pmin"]) ? -999.99 : data["gen"][i]["Pmin"]
    if data["case_format"] == "MATPOWER"
      generators[i].Pc1 = data["gen"][i]["Pc1"]
      generators[i].Pc2 = data["gen"][i]["Pc2"]
      generators[i].Qc1min = data["gen"][i]["Qc1min"]
      generators[i].Qc1max = data["gen"][i]["Qc1max"]
      generators[i].Qc2min = data["gen"][i]["Qc2min"]
      generators[i].Qc2max = data["gen"][i]["Qc2max"]
    end
    uc_initialize(generators[i])

    generators[i].gentype = data["gencost"][i]["cost_type"]
    generators[i].startup = data["gencost"][i]["startup"]
    generators[i].shutdown = data["gencost"][i]["shutdown"]
    generators[i].n = data["gencost"][i]["n"]
    @assert generators[i].gentype == 2 && generators[i].n == 3
    generators[i].coeff = [data["gencost"][i]["c2"], data["gencost"][i]["c1"], data["gencost"][i]["c0"]]
  end

  # Read load profiles
  load = get_load(load_prefix)

  # build a dictionary between buses ids and their indexes
  busIdx = mapBusIdToIdx(buses)

  # set up the FromLines and ToLines for each bus
  FromLines,ToLines = mapLinesToBuses(buses, lines, busIdx)

  # generators at each bus
  BusGeners = mapGenersToBuses(buses, generators, busIdx)

  return UCOPFData(buses, lines, load, generators, bus_ref, data["baseMVA"], busIdx, FromLines, ToLines, BusGeners)
end

# TODO: implement this later
#=
function uc_opf_loaddata_dlm(case_name, load_prefix, lineOff=Line(); VI=Array{Int}, VD=Array{Float64})
end
=#

function uc_opf_loaddata(case_name, load_prefix; VI=Array{Int}, VD=Array{Float64}, case_format="MATPOWER")
  format = lowercase(case_format)
  if format in ["matpower", "pglib"]
    return uc_opf_loaddata_matpower(case_name, load_prefix; VI=VI, VD=VD, case_format=format)
  else
    # return uc_opf_loaddata_dlm(case_name, load_prefix; VI=VI, VD=VD)
  end
end

function get_generator_data(data::UCOPFData; use_gpu=false)
  ngen = length(data.generators)

  if use_gpu
      pgmin = CuArray{Float64}(undef, ngen)
      pgmax = CuArray{Float64}(undef, ngen)
      qgmin = CuArray{Float64}(undef, ngen)
      qgmax = CuArray{Float64}(undef, ngen)
      ru = CuArray{Float64}(undef, ngen)
      rd = CuArray{Float64}(undef, ngen)
      minUp = CuArray{Int}(undef, ngen)
      minDown = CuArray{Int}(undef, ngen)
      c2 = CuArray{Float64}(undef, ngen)
      c1 = CuArray{Float64}(undef, ngen)
      c0 = CuArray{Float64}(undef, ngen)
  else
      pgmin = Array{Float64}(undef, ngen)
      pgmax = Array{Float64}(undef, ngen)
      qgmin = Array{Float64}(undef, ngen)
      qgmax = Array{Float64}(undef, ngen)
      ru = Array{Float64}(undef, ngen)
      rd = Array{Float64}(undef, ngen)
      minUp = Array{Int}(undef, ngen)
      minDown = Array{Int}(undef, ngen)
      c2 = Array{Float64}(undef, ngen)
      c1 = Array{Float64}(undef, ngen)
      c0 = Array{Float64}(undef, ngen)
  end

  Pmin = Float64[data.generators[g].Pmin for g in 1:ngen]
  Pmax = Float64[data.generators[g].Pmax for g in 1:ngen]
  Qmin = Float64[data.generators[g].Qmin for g in 1:ngen]
  Qmax = Float64[data.generators[g].Qmax for g in 1:ngen]
  Ru = Float64[data.generators[g].ru for g in 1:ngen]
  Rd = Float64[data.generators[g].rd for g in 1:ngen]
  MinUp = Int[data.generators[g].minUp for g in 1:ngen]
  MinDown = Int[data.generators[g].minDown for g in 1:ngen]
  coeff0 = Float64[data.generators[g].coeff[3] for g in 1:ngen]
  coeff1 = Float64[data.generators[g].coeff[2] for g in 1:ngen]
  coeff2 = Float64[data.generators[g].coeff[1] for g in 1:ngen]
  copyto!(pgmin, Pmin)
  copyto!(pgmax, Pmax)
  copyto!(qgmin, Qmin)
  copyto!(qgmax, Qmax)
  copyto!(ru, Ru)
  copyto!(rd, Rd)
  copyto!(minUp, MinUp)
  copyto!(minDown, MinDown)
  copyto!(c0, coeff0)
  copyto!(c1, coeff1)
  copyto!(c2, coeff2)

  return pgmin,pgmax,qgmin,qgmax,c2,c1,c0,ru,rd,minUp,minDown
end

function get_bus_data(data::UCOPFData; use_gpu=false)
  nbus = length(data.buses)

  FrIdx = Int[l for b=1:nbus for l in data.FromLines[b]]
  ToIdx = Int[l for b=1:nbus for l in data.ToLines[b]]
  GenIdx = Int[g for b=1:nbus for g in data.BusGenerators[b]]
  FrStart = accumulate(+, vcat([1], [length(data.FromLines[b]) for b=1:nbus]))
  ToStart = accumulate(+, vcat([1], [length(data.ToLines[b]) for b=1:nbus]))
  GenStart = accumulate(+, vcat([1], [length(data.BusGenerators[b]) for b=1:nbus]))

  # Pd = Float64[j for i=1:nbus for j in data.buses[i].Pd]
  # Qd = Float64[j for i=1:nbus for j in data.buses[i].Pd]
  Vmin = Float64[data.buses[i].Vmin for i=1:nbus]
  Vmax = Float64[data.buses[i].Vmax for i=1:nbus]

  if use_gpu
      cuFrIdx = CuArray{Int}(undef, length(FrIdx))
      cuToIdx = CuArray{Int}(undef, length(ToIdx))
      cuGenIdx = CuArray{Int}(undef, length(GenIdx))
      cuFrStart = CuArray{Int}(undef, length(FrStart))
      cuToStart = CuArray{Int}(undef, length(ToStart))
      cuGenStart = CuArray{Int}(undef, length(GenStart))
      # cuPd = CuArray{Float64}(undef, nbus*T)
      # cuQd = CuArray{Float64}(undef, nbus*T)
      cuVmax = CuArray{Float64}(undef, nbus)
      cuVmin = CuArray{Float64}(undef, nbus)

      copyto!(cuFrIdx, FrIdx)
      copyto!(cuToIdx, ToIdx)
      copyto!(cuGenIdx, GenIdx)
      copyto!(cuFrStart, FrStart)
      copyto!(cuToStart, ToStart)
      copyto!(cuGenStart, GenStart)
      # copyto!(cuPd, Pd)
      # copyto!(cuQd, Qd)
      copyto!(cuVmax, Vmax)
      copyto!(cuVmin, Vmin)

      # return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuPd,cuQd,cuVmin,cuVmax
      return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuVmin,cuVmax
  else
      # return FrStart,FrIdx,ToStart,ToIdx,GenStart,GenIdx,Pd,Qd,Vmin,Vmax
      return FrStart,FrIdx,ToStart,ToIdx,GenStart,GenIdx,Vmin,Vmax
  end
end

function get_branch_data(data::UCOPFData; use_gpu::Bool=false, tight_factor::Float64=1.0)
  buses = data.buses
  lines = data.lines
  BusIdx = data.BusIdx
  nline = length(data.lines)
  ybus = Ybus{Array{Float64}}(computeAdmitances(data.lines, data.buses, data.baseMVA; VI=Array{Int}, VD=Array{Float64})...)
  frVmBound = Float64[ x for l=1:nline for x in (buses[BusIdx[lines[l].from]].Vmin, buses[BusIdx[lines[l].from]].Vmax) ]
  toVmBound = Float64[ x for l=1:nline for x in (buses[BusIdx[lines[l].to]].Vmin, buses[BusIdx[lines[l].to]].Vmax) ]
  frVaBound = Float64[ x for l=1:nline for x in (-2*pi,2*pi) ]
  toVaBound = Float64[ x for l=1:nline for x in (-2*pi,2*pi) ]
  for l=1:nline
      if BusIdx[lines[l].from] == data.bus_ref
          frVaBound[2*l-1] = 0.0
          frVaBound[2*l] = 0.0
      end
      if BusIdx[lines[l].to] == data.bus_ref
          toVaBound[2*l-1] = 0.0
          toVaBound[2*l] = 0.0
      end
  end
  rateA = [ data.lines[l].rateA == 0.0 ? 1e3 : tight_factor*(data.lines[l].rateA / data.baseMVA)^2 for l=1:nline ]

  if use_gpu
    cuYshR = CuArray{Float64}(undef, length(ybus.YshR))
    cuYshI = CuArray{Float64}(undef, length(ybus.YshI))
    cuYffR = CuArray{Float64}(undef, nline)
    cuYffI = CuArray{Float64}(undef, nline)
    cuYftR = CuArray{Float64}(undef, nline)
    cuYftI = CuArray{Float64}(undef, nline)
    cuYttR = CuArray{Float64}(undef, nline)
    cuYttI = CuArray{Float64}(undef, nline)
    cuYtfR = CuArray{Float64}(undef, nline)
    cuYtfI = CuArray{Float64}(undef, nline)
    cuFrVmBound = CuArray{Float64}(undef, 2*nline)
    cuToVmBound = CuArray{Float64}(undef, 2*nline)
    cuFrVaBound = CuArray{Float64}(undef, 2*nline)
    cuToVaBound = CuArray{Float64}(undef, 2*nline)
    cuRateA = CuArray{Float64}(undef, nline)
    copyto!(cuYshR, ybus.YshR)
    copyto!(cuYshI, ybus.YshI)
    copyto!(cuYffR, ybus.YffR)
    copyto!(cuYffI, ybus.YffI)
    copyto!(cuYftR, ybus.YftR)
    copyto!(cuYftI, ybus.YftI)
    copyto!(cuYttR, ybus.YttR)
    copyto!(cuYttI, ybus.YttI)
    copyto!(cuYtfR, ybus.YtfR)
    copyto!(cuYtfI, ybus.YtfI)
    copyto!(cuFrVmBound, frVmBound)
    copyto!(cuToVmBound, toVmBound)
    copyto!(cuFrVaBound, frVaBound)
    copyto!(cuToVaBound, toVaBound)
    copyto!(cuRateA, rateA)

    return cuYshR, cuYshI, cuYffR, cuYffI, cuYftR, cuYftI,
           cuYttR, cuYttI, cuYtfR, cuYtfI, cuFrVmBound, cuToVmBound,
           cuFrVaBound, cuToVaBound, cuRateA
  else
    return ybus.YshR, ybus.YshI, ybus.YffR, ybus.YffI, ybus.YftR, ybus.YftI,
           ybus.YttR, ybus.YttI, ybus.YtfR, ybus.YtfI, frVmBound, toVmBound,
           frVaBound, toVaBound, rateA
  end
end

function get_branch_bus_index(data::UCOPFData; use_gpu=false)
  lines = data.lines
  BusIdx = data.BusIdx
  nline = length(lines)

  brBusIdx = Int[ x for l=1:nline for x in (BusIdx[lines[l].from], BusIdx[lines[l].to]) ]

  if use_gpu
      cu_brBusIdx = CuArray{Int}(undef, 2*nline)
      copyto!(cu_brBusIdx, brBusIdx)
      return cu_brBusIdx
  else
      return brBusIdx
  end
end