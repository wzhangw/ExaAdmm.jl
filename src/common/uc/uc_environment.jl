mutable struct UCAdmmEnv{T,TD,TI,TM} <: AbstractAdmmEnv{T,TD,TI,TM}
    case::String
    data::UCOPFData
    load::Load{TM}
    initial_rho_pq::Float64
    initial_rho_va::Float64
    tight_factor::Float64
    horizon_length::Int
    use_gpu::Bool
    use_linelimit::Bool
    use_twolevel::Bool
    use_mpi::Bool
    load_specified::Bool
    gpu_no::Int
    comm::MPI.Comm

    params::Parameters

    function UCAdmmEnv{T,TD,TI,TM}(
        case::String, rho_pq::Float64, rho_va::Float64;
        case_format="matpower",
        use_gpu=false, use_linelimit=false, use_twolevel=false, use_mpi=false,
        gpu_no::Int=1, verbose::Int=1, tight_factor=1.0,
        horizon_length=1, load_prefix::String="", comm::MPI.Comm=MPI.COMM_WORLD
    ) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        env = new{T,TD,TI,TM}()
        env.case = case
        env.data = uc_opf_loaddata(env.case, load_prefix; VI=TI, VD=TD, case_format=case_format)
        env.initial_rho_pq = rho_pq
        env.initial_rho_va = rho_va
        env.tight_factor = tight_factor
        env.use_gpu = use_gpu
        env.use_linelimit = use_linelimit
        env.use_mpi = use_mpi
        env.gpu_no = gpu_no
        env.use_twolevel = use_twolevel
        env.load_specified = false
        env.comm = comm

        env.params = Parameters()
        env.params.verbose = verbose

        env.horizon_length = horizon_length
        # if !isempty(load_prefix)
        #     env.load = get_load(load_prefix; use_gpu=use_gpu)
        #     @assert size(env.load.pd) == size(env.load.qd)
        #     @assert size(env.load.pd,2) >= horizon_length && size(env.load.qd,2) >= horizon_length
        #     env.load_specified = true
        # end
        env.load = env.data.load
        env.load_specified = true

        return env
    end
end

mutable struct UCModel{T,TD,TI,TM} <: AbstractOPFModel{T,TD,TI,TM}
    solution::AbstractSolution{T,TD}

    t::Int
    n::Int
    ngen::Int
    nline::Int
    nbus::Int
    nvar::Int

    gen_start::Int
    line_start::Int

    baseMVA::T
    pgmin::TD
    pgmax::TD
    qgmin::TD
    qgmax::TD
    pgmin_curr::TD   # taking ramping into account for rolling horizon
    pgmax_curr::TD   # taking ramping into account for rolling horizon
    ramp_rate::TD
    c2::TD
    c1::TD
    c0::TD
    ru::TD
    rd::TD
    minUp::TI
    minDown::TI
    YshR::TD
    YshI::TD
    YffR::TD
    YffI::TD
    YftR::TD
    YftI::TD
    YttR::TD
    YttI::TD
    YtfR::TD
    YtfI::TD
    FrVmBound::TD
    ToVmBound::TD
    FrVaBound::TD
    ToVaBound::TD
    rateA::TD
    FrStart::TI
    FrIdx::TI
    ToStart::TI
    ToIdx::TI
    GenStart::TI
    GenIdx::TI
    Pd::TD
    Qd::TD
    Vmin::TD
    Vmax::TD

    membuf::TM

    # Two-Level ADMM
    nvar_u::Int
    nvar_v::Int
    bus_start::Int # this is for varibles of type v.
    brBusIdx::TI

    # Padded sizes for MPI
    nline_padded::Int
    nvar_u_padded::Int
    nvar_padded::Int

    function UCModel{T,TD,TI,TM}(env::UCAdmmEnv{T,TD,TI,TM}; ramp_ratio=0.2) where {T, TD<:AbstractArray{T}, TI<:AbstractArray{Int}, TM<:AbstractArray{T,2}}
        model = new{T,TD,TI,TM}()

        model.baseMVA = env.data.baseMVA
        model.n = (env.use_linelimit == true) ? 6 : 4
        model.ngen = length(env.data.generators)
        model.nline = length(env.data.lines)
        model.nbus = length(env.data.buses)
        model.nline_padded = model.nline
        model.t = 2

        # Memory space is padded for the lines as a multiple of # processes.
        if env.use_mpi
            nprocs = MPI.Comm_size(env.comm)
            model.nline_padded = nprocs * div(model.nline, nprocs, RoundUp)
        end

        model.nvar = 2*model.ngen*model.t + 8*model.nline*model.t
        model.nvar_padded = model.nvar + 8*(model.nline_padded - model.nline)
        model.gen_start = 1
        model.line_start = 2*model.ngen*model.t + 1
        model.pgmin, model.pgmax, model.qgmin, model.qgmax, model.c2, model.c1, model.c0, model.ru, model.rd, model.minUp, model.minDown = get_generator_data(env.data; use_gpu=env.use_gpu)
        model.YshR, model.YshI, model.YffR, model.YffI, model.YftR, model.YftI,
            model.YttR, model.YttI, model.YtfR, model.YtfI,
            model.FrVmBound, model.ToVmBound,
            model.FrVaBound, model.ToVaBound, model.rateA = get_branch_data(env.data; use_gpu=env.use_gpu, tight_factor=env.tight_factor)
        model.FrStart, model.FrIdx, model.ToStart, model.ToIdx, model.GenStart, model.GenIdx, model.Pd, model.Qd, model.Vmin, model.Vmax = get_bus_data(env.data; use_gpu=env.use_gpu)
        model.brBusIdx = get_branch_bus_index(env.data; use_gpu=env.use_gpu)

        model.pgmin_curr = TD(undef, model.ngen)
        model.pgmax_curr = TD(undef, model.ngen)
        copyto!(model.pgmin_curr, model.pgmin)
        copyto!(model.pgmax_curr, model.pgmax)

        model.ramp_rate = TD(undef, model.ngen)
        model.ramp_rate .= ramp_ratio.*model.pgmax

        if env.params.obj_scale != 1.0
            model.c2 .*= env.params.obj_scale
            model.c1 .*= env.params.obj_scale
            model.c0 .*= env.params.obj_scale
        end

        # These are only for two-level ADMM.
        model.nvar_u = 2*model.ngen*model.t + 8*model.nline*model.t
        model.nvar_u_padded = model.nvar_u + 8*(model.nline_padded - model.nline)
        model.nvar_v = 2*model.ngen*model.t + 4*model.nline*model.t + 2*model.nbus*model.t
        model.bus_start = 2*model.ngen*model.t + 4*model.nline*model.t + 1
        if env.use_twolevel
            model.nvar = model.nvar_u + model.nvar_v
            model.nvar_padded = model.nvar_u_padded + model.nvar_v
        end

        # Memory space is allocated based on the padded size.
        model.solution = ifelse(env.use_twolevel,
            SolutionTwoLevel{T,TD}(model.nvar_padded, model.nvar_v, model.nline_padded),
            SolutionOneLevel{T,TD}(model.nvar_padded))
        init_solution!(model, model.solution, env.initial_rho_pq, env.initial_rho_va)

        model.membuf = TM(undef, (31, model.nline*model.t))
        fill!(model.membuf, 0.0)

        return model
    end
end
