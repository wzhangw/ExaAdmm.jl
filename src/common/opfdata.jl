mutable struct Bus
  bus_i::Int
  bustype::Int
  Pd::Float64
  Qd::Float64
  Gs::Float64
  Bs::Float64
  area::Int
  Vm::Float64
  Va::Float64
  baseKV::Float64
  zone::Int
  Vmax::Float64
  Vmin::Float64
end

mutable struct Line
  from::Int
  to::Int
  r::Float64
  x::Float64
  b::Float64
  rateA::Float64
  rateB::Float64
  rateC::Float64
  ratio::Float64 #TAP
  angle::Float64 #SHIFT
  status::Int
  angmin::Float64
  angmax::Float64
end
Line() = Line(0,0,0.,0.,0.,0.,0.,0.,0.,0.,0,0.,0.)

mutable struct Gener
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
  # .gencost fields
  gentype::Int
  startup::Float64
  shutdown::Float64
  n::Int
  coeff::Array
end

struct OPFData
  buses::Array{Bus}
  lines::Array{Line}
  generators::Array{Gener}
  bus_ref::Int
  baseMVA::Float64
  BusIdx::Dict{Int,Int}    #map from bus ID to bus index
  FromLines::Array         #From lines for each bus (Array of Array)
  ToLines::Array           #To lines for each bus (Array of Array)
  BusGenerators::Array     #list of generators for each bus (Array of Array)
end

mutable struct Ybus{VD}
  YffR::VD
  YffI::VD
  YttR::VD
  YttI::VD
  YftR::VD
  YftI::VD
  YtfR::VD
  YtfI::VD
  YshR::Array{Float64}
  YshI::Array{Float64}

  Ybus{VD}(YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI) where {VD} = new(YffR,YffI,YttR,YttI,YftR,YftI,YtfR,YtfI,YshR,YshI)
end

mutable struct Load{TM}
  pd::TM
  qd::TM

  function Load{TM}(pd_mat, qd_mat) where {TM}
    m, n = size(pd_mat)
    mq, nq = size(qd_mat)
    @assert m == mq && n == nq

    load = new{TM}()
    load.pd = TM(undef, (m,n))
    load.qd = TM(undef, (m,n))
    copyto!(load.pd, pd_mat)
    copyto!(load.qd, qd_mat)

    return load
  end
end

function get_load(name::String; load_scale=1.0, use_gpu=false)
  pd_mat = readdlm(name*".Pd")
  qd_mat = readdlm(name*".Qd")
  if use_gpu
    load = Load{CuArray{Float64,2}}(pd_mat.*load_scale, qd_mat.*load_scale)
  else
    load = Load{Array{Float64,2}}(pd_mat.*load_scale, qd_mat.*load_scale)
  end
  return load
end

function opf_loaddata_matpower(case_name, lineOff=Line(); VI=Array{Int}, VD=Array{Float64}, case_format="matpower")
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
  generators = Array{Gener}(undef, ngen)
  for i=1:ngen
    generators[i] = Gener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, Array{Int}(undef, 0)) #gen_arr[i,1:end]...)
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

    generators[i].gentype = data["gencost"][i]["cost_type"]
    generators[i].startup = data["gencost"][i]["startup"]
    generators[i].shutdown = data["gencost"][i]["shutdown"]
    generators[i].n = data["gencost"][i]["n"]
    @assert generators[i].gentype == 2 && generators[i].n == 3
    generators[i].coeff = [data["gencost"][i]["c2"], data["gencost"][i]["c1"], data["gencost"][i]["c0"]]
  end

  # build a dictionary between buses ids and their indexes
  busIdx = mapBusIdToIdx(buses)

  # set up the FromLines and ToLines for each bus
  FromLines,ToLines = mapLinesToBuses(buses, lines, busIdx)

  # generators at each bus
  BusGeners = mapGenersToBuses(buses, generators, busIdx)

  return OPFData(buses, lines, generators, bus_ref, data["baseMVA"], busIdx, FromLines, ToLines, BusGeners)
end

function opf_loaddata_dlm(case_name, lineOff=Line(); VI=Array{Int}, VD=Array{Float64})
  #
  # load buses
  #
  # bus_arr = readdlm("data/" * case_name * ".bus")
  bus_arr = readdlm(case_name * ".bus")
  num_buses = size(bus_arr,1)
  buses = Array{Bus}(undef, num_buses)
  bus_ref=-1
  for i in 1:num_buses
    @assert bus_arr[i,1]>0  #don't support nonpositive bus ids
    buses[i] = Bus(bus_arr[i,1:13]...)
    buses[i].Va *= pi/180
    if buses[i].bustype==3
      if bus_ref>0
        error("More than one reference bus present in the data")
      else
         bus_ref=i
      end
    end
    #println("bus ", i, " ", buses[i].Vmin, "      ", buses[i].Vmax)
  end

  #
  # load branches/lines
  #
  # branch_arr = readdlm("data/" * case_name * ".branch")
  branch_arr = readdlm(case_name * ".branch")
  num_lines = size(branch_arr,1)
  lines_on = findall((branch_arr[:,11].>0) .& ((branch_arr[:,1].!=lineOff.from) .| (branch_arr[:,2].!=lineOff.to)) )
  num_on   = length(lines_on)

  if lineOff.from>0 && lineOff.to>0
    println("opf_loaddata: was asked to remove line from,to=", lineOff.from, ",", lineOff.to)
    #println(lines_on, branch_arr[:,1].!=lineOff.from, branch_arr[:,2].!=lineOff.to)
  end
  if length(findall(branch_arr[:,11].==0))>0
    println("opf_loaddata: ", num_lines-length(findall(branch_arr[:,11].>0)), " lines are off and will be discarded (out of ", num_lines, ")")
  end



  lines = Array{Line}(undef, num_on)

  lit=0
  for i in lines_on
    @assert branch_arr[i,11] == 1  #should be on since we discarded all other
    lit += 1
    lines[lit] = Line(branch_arr[i, 1:13]...)
    #=
    if (lines[lit].angmin != 0 || lines[lit].angmax != 0) && (lines[lit].angmin>-360 || lines[lit].angmax<360)
      println("Voltage bounds on line ", i, " with angmin ", lines[lit].angmin, " and angmax ", lines[lit].angmax)
      error("Bounds of voltage angles are still to be implemented.")
    end
    =#

  end
  @assert lit == num_on

  #
  # load generators
  #
  # gen_arr = readdlm("data/" * case_name * ".gen")
  gen_arr = readdlm(case_name * ".gen")
  # costgen_arr = readdlm("data/" * case_name * ".gencost")
  costgen_arr = readdlm(case_name * ".gencost")
  num_gens = size(gen_arr,1)

  baseMVA=100

  @assert num_gens == size(costgen_arr,1)

  gens_on=findall(x->x!=0, gen_arr[:,8]); num_on=length(gens_on)
  if num_gens-num_on>0
    println("loaddata: ", num_gens-num_on, " generators are off and will be discarded (out of ", num_gens, ")")
  end

  generators = Array{Gener}(undef, num_on)
  i=0
  for git in gens_on
    i += 1
    generators[i] = Gener(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0, Array{Int}(undef, 0)) #gen_arr[i,1:end]...)

    generators[i].bus      = gen_arr[git,1]
    generators[i].Pg       = gen_arr[git,2] / baseMVA
    generators[i].Qg       = gen_arr[git,3] / baseMVA
    generators[i].Qmax     = gen_arr[git,4] / baseMVA
    generators[i].Qmin     = gen_arr[git,5] / baseMVA
    generators[i].Vg       = gen_arr[git,6]
    generators[i].mBase    = gen_arr[git,7]
    generators[i].status   = gen_arr[git,8]
    @assert generators[i].status==1
    generators[i].Pmax     = gen_arr[git,9]  / baseMVA
    generators[i].Pmin     = gen_arr[git,10] / baseMVA
    generators[i].Pc1      = gen_arr[git,11]
    generators[i].Pc2      = gen_arr[git,12]
    generators[i].Qc1min   = gen_arr[git,13]
    generators[i].Qc1max   = gen_arr[git,14]
    generators[i].Qc2min   = gen_arr[git,15]
    generators[i].Qc2max   = gen_arr[git,16]
    generators[i].gentype  = costgen_arr[git,1]
    generators[i].startup  = costgen_arr[git,2]
    generators[i].shutdown = costgen_arr[git,3]
    generators[i].n        = costgen_arr[git,4]
    @assert(generators[i].n <= 3 && generators[i].n >= 2)
    if generators[i].gentype == 1
      generators[i].coeff = costgen_arr[git,5:end]
      error("Piecewise linear costs remains to be implemented.")
    else
      if generators[i].gentype == 2
        generators[i].coeff = costgen_arr[git,5:end]
        #println(generators[i].coeff, " ", length(generators[i].coeff), " ", generators[i].coeff[2])
      else
        error("Invalid generator cost model in the data.")
      end
    end
  end

  # build a dictionary between buses ids and their indexes
  busIdx = mapBusIdToIdx(buses)

  # set up the FromLines and ToLines for each bus
  FromLines,ToLines = mapLinesToBuses(buses, lines, busIdx)

  # generators at each bus
  BusGeners = mapGenersToBuses(buses, generators, busIdx)

  #println(generators)
  #println(bus_ref)
  return OPFData(buses, lines, generators, bus_ref, baseMVA, busIdx, FromLines, ToLines, BusGeners)
end

function opf_loaddata(case_name; VI=Array{Int}, VD=Array{Float64}, case_format="MATPOWER")
  format = lowercase(case_format)
  if format in ["matpower", "pglib"]
    return opf_loaddata_matpower(case_name; VI=VI, VD=VD, case_format=format)
  else
    return opf_loaddata_dlm(case_name; VI=VI, VD=VD)
  end
end

function  computeAdmitances(lines, buses, baseMVA; VI=Array{Int}, VD=Array{Float64})
  nlines = length(lines)
  YffR=Array{Float64}(undef, nlines)
  YffI=Array{Float64}(undef, nlines)
  YttR=Array{Float64}(undef, nlines)
  YttI=Array{Float64}(undef, nlines)
  YftR=Array{Float64}(undef, nlines)
  YftI=Array{Float64}(undef, nlines)
  YtfR=Array{Float64}(undef, nlines)
  YtfI=Array{Float64}(undef, nlines)

  for i in 1:nlines
    @assert lines[i].status == 1
    Ys = 1/(lines[i].r + lines[i].x*im)
    #assign nonzero tap ratio
    tap = lines[i].ratio==0 ? 1.0 : lines[i].ratio

    #add phase shifters
    tap *= exp(lines[i].angle * pi/180 * im)

    Ytt = Ys + lines[i].b/2*im
    Yff = Ytt / (tap*conj(tap))
    Yft = -Ys / conj(tap)
    Ytf = -Ys / tap

    #split into real and imag parts
    YffR[i] = real(Yff); YffI[i] = imag(Yff)
    YttR[i] = real(Ytt); YttI[i] = imag(Ytt)
    YtfR[i] = real(Ytf); YtfI[i] = imag(Ytf)
    YftR[i] = real(Yft); YftI[i] = imag(Yft)
    #@printf("[%4d]  tap=%12.9f   %12.9f\n", i, real(tap), imag(tap));
  end

  nbuses = length(buses)
  YshR = Array{Float64}(undef, nbuses)
  YshI = Array{Float64}(undef, nbuses)
  for i in 1:nbuses
    YshR[i] = buses[i].Gs / baseMVA
    YshI[i] = buses[i].Bs / baseMVA
    #@printf("[%4d]   Ysh  %15.12f + %15.12f i \n", i, YshR[i], YshI[i])
  end

  @assert 0==length(findall(isnan.(YffR)))+length(findall(isinf.(YffR)))
  @assert 0==length(findall(isnan.(YffI)))+length(findall(isinf.(YffI)))
  @assert 0==length(findall(isnan.(YttR)))+length(findall(isinf.(YttR)))
  @assert 0==length(findall(isnan.(YttI)))+length(findall(isinf.(YttI)))
  @assert 0==length(findall(isnan.(YftR)))+length(findall(isinf.(YftR)))
  @assert 0==length(findall(isnan.(YftI)))+length(findall(isinf.(YftI)))
  @assert 0==length(findall(isnan.(YtfR)))+length(findall(isinf.(YtfR)))
  @assert 0==length(findall(isnan.(YtfI)))+length(findall(isinf.(YtfI)))
  @assert 0==length(findall(isnan.(YshR)))+length(findall(isinf.(YshR)))
  @assert 0==length(findall(isnan.(YshI)))+length(findall(isinf.(YshI)))

  if isa(VD, CuArray)
    return copyto!(VD(undef, nlines), 1, YffR, 1, nlines),
           copyto!(VD(undef, nlines), 1, YffI, 1, nlines),
           copyto!(VD(undef, nlines), 1, YttR, 1, nlines),
           copyto!(VD(undef, nlines), 1, YttI, 1, nlines),
           copyto!(VD(undef, nlines), 1, YftR, 1, nlines),
           copyto!(VD(undef, nlines), 1, YftI, 1, nlines),
           copyto!(VD(undef, nlines), 1, YtfR, 1, nlines),
           copyto!(VD(undef, nlines), 1, YtfI, 1, nlines),
           YshR, YshI
  else
    return YffR, YffI, YttR, YttI, YftR, YftI, YtfR, YtfI, YshR, YshI
  end
end


# Builds a map from lines to buses.
# For each line we store an array with zero or one element containing
# the  'From' and 'To'  bus number.
function mapLinesToBuses(buses, lines, busDict)
  nbus = length(buses)
  FromLines = [Int[] for b in 1:nbus]
  ToLines   = [Int[] for b in 1:nbus]
  for i in 1:length(lines)
    busID = busDict[lines[i].from]
    @assert 1<= busID <= nbus
    push!(FromLines[busID], i)

    busID = busDict[lines[i].to]
    @assert 1<= busID  <= nbus
    push!(ToLines[busID], i)
  end

  return FromLines,ToLines
end

# Builds a mapping between bus ids and bus indexes
#
# Returns a dictionary with bus ids as keys and bus indexes as values
function mapBusIdToIdx(buses)
  dict = Dict{Int,Int}()
  for b in 1:length(buses)
    @assert !haskey(dict,buses[b].bus_i)
    dict[buses[b].bus_i] = b
  end
  return dict
end


# Builds a map between buses and generators.
# For each bus we keep an array of corresponding generators number (as array).
#
# (Can be more than one generator per bus)
function mapGenersToBuses(buses, generators,busDict)
  gen2bus = [Int[] for b in 1:length(buses)]
  for g in 1:length(generators)
    busID = busDict[ generators[g].bus ]
    #@assert(0==length(gen2bus[busID])) #at most one generator per bus
    push!(gen2bus[busID], g)
  end
  return gen2bus
end

function get_generator_data(data::OPFData; use_gpu=false)
  ngen = length(data.generators)

  if use_gpu
      pgmin = CuArray{Float64}(undef, ngen)
      pgmax = CuArray{Float64}(undef, ngen)
      qgmin = CuArray{Float64}(undef, ngen)
      qgmax = CuArray{Float64}(undef, ngen)
      c2 = CuArray{Float64}(undef, ngen)
      c1 = CuArray{Float64}(undef, ngen)
      c0 = CuArray{Float64}(undef, ngen)
  else
      pgmin = Array{Float64}(undef, ngen)
      pgmax = Array{Float64}(undef, ngen)
      qgmin = Array{Float64}(undef, ngen)
      qgmax = Array{Float64}(undef, ngen)
      c2 = Array{Float64}(undef, ngen)
      c1 = Array{Float64}(undef, ngen)
      c0 = Array{Float64}(undef, ngen)
  end

  Pmin = Float64[data.generators[g].Pmin for g in 1:ngen]
  Pmax = Float64[data.generators[g].Pmax for g in 1:ngen]
  Qmin = Float64[data.generators[g].Qmin for g in 1:ngen]
  Qmax = Float64[data.generators[g].Qmax for g in 1:ngen]
  coeff0 = Float64[data.generators[g].coeff[3] for g in 1:ngen]
  coeff1 = Float64[data.generators[g].coeff[2] for g in 1:ngen]
  coeff2 = Float64[data.generators[g].coeff[1] for g in 1:ngen]
  copyto!(pgmin, Pmin)
  copyto!(pgmax, Pmax)
  copyto!(qgmin, Qmin)
  copyto!(qgmax, Qmax)
  copyto!(c0, coeff0)
  copyto!(c1, coeff1)
  copyto!(c2, coeff2)

  return pgmin,pgmax,qgmin,qgmax,c2,c1,c0
end

function get_bus_data(data::OPFData; use_gpu=false)
  nbus = length(data.buses)

  FrIdx = Int[l for b=1:nbus for l in data.FromLines[b]]
  ToIdx = Int[l for b=1:nbus for l in data.ToLines[b]]
  GenIdx = Int[g for b=1:nbus for g in data.BusGenerators[b]]
  FrStart = accumulate(+, vcat([1], [length(data.FromLines[b]) for b=1:nbus]))
  ToStart = accumulate(+, vcat([1], [length(data.ToLines[b]) for b=1:nbus]))
  GenStart = accumulate(+, vcat([1], [length(data.BusGenerators[b]) for b=1:nbus]))

  Pd = Float64[data.buses[i].Pd for i=1:nbus]
  Qd = Float64[data.buses[i].Qd for i=1:nbus]
  Vmin = Float64[data.buses[i].Vmin for i=1:nbus]
  Vmax = Float64[data.buses[i].Vmax for i=1:nbus]

  if use_gpu
      cuFrIdx = CuArray{Int}(undef, length(FrIdx))
      cuToIdx = CuArray{Int}(undef, length(ToIdx))
      cuGenIdx = CuArray{Int}(undef, length(GenIdx))
      cuFrStart = CuArray{Int}(undef, length(FrStart))
      cuToStart = CuArray{Int}(undef, length(ToStart))
      cuGenStart = CuArray{Int}(undef, length(GenStart))
      cuPd = CuArray{Float64}(undef, nbus)
      cuQd = CuArray{Float64}(undef, nbus)
      cuVmax = CuArray{Float64}(undef, nbus)
      cuVmin = CuArray{Float64}(undef, nbus)

      copyto!(cuFrIdx, FrIdx)
      copyto!(cuToIdx, ToIdx)
      copyto!(cuGenIdx, GenIdx)
      copyto!(cuFrStart, FrStart)
      copyto!(cuToStart, ToStart)
      copyto!(cuGenStart, GenStart)
      copyto!(cuPd, Pd)
      copyto!(cuQd, Qd)
      copyto!(cuVmax, Vmax)
      copyto!(cuVmin, Vmin)

      return cuFrStart,cuFrIdx,cuToStart,cuToIdx,cuGenStart,cuGenIdx,cuPd,cuQd,cuVmin,cuVmax
  else
      return FrStart,FrIdx,ToStart,ToIdx,GenStart,GenIdx,Pd,Qd,Vmin,Vmax
  end
end

function get_branch_data(data::OPFData; use_gpu::Bool=false, tight_factor::Float64=1.0)
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

function get_branch_bus_index(data::OPFData; use_gpu=false)
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