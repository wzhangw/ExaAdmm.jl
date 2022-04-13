function uc_set_rateA_kernel(nline::Int, t::Int, param::CuDeviceArray{Float64,2}, rateA::CuDeviceArray{Float64,1})
    tx = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    l = div(tx, t, RoundUp)

    if tx <= nline*t
        param[29,tx] = (rateA[l] == 0.0) ? 1e3 : rateA[l]
    end

    return
end

function uc_admm_restart(
    env::UCAdmmEnv{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}},
    mod::UCModel{Float64,CuArray{Float64,1},CuArray{Int,1},CuArray{Float64,2}}
)
    data, par, sol = env.data, env.params, mod.solution

    shift_lines = 0
    shmem_size = sizeof(Float64)*(14*mod.n+3*mod.n^2) + sizeof(Int)*(4*mod.n)

    nblk_gen = div(mod.ngen, 32, RoundUp)
    nblk_br = mod.nline*mod.t
    nblk_bus = div(mod.nbus*mod.t, 32, RoundUp)

    it = 0
    time_gen = time_br = time_bus = 0.0

    beta = 1e3
    c = 6.0
    theta = 0.8
    sqrt_d = sqrt(mod.nvar)
    OUTER_TOL = sqrt_d*(par.outer_eps)

    outer = inner = cumul = 0
    mismatch = Inf
    z_prev_norm = z_curr_norm = Inf

    overall_time = @timed begin
    while outer < par.outer_iterlim
        outer += 1

        CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1,64)+1) copy_data_kernel(mod.nvar, sol.z_outer, sol.z_curr)
        z_prev_norm = CUDA.norm(sol.z_curr)

        inner = 0
        while inner < par.inner_iterlim
            inner += 1
            cumul += 1

            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) copy_data_kernel(mod.nvar, sol.z_prev, sol.z_curr)

            tgpu = uc_generator_kernel_two_level(mod, mod.baseMVA, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho)
            time_gen += tgpu.time

            if env.use_linelimit
                tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size uc_auglag_linelimit_two_level_alternative(
                                                    mod.n, mod.nline, mod.t, mod.line_start,
                                                    inner, par.max_auglag, par.mu_max, par.scale,
                                                    sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                                                    shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                                                    mod.YttR, mod.YttI, mod.YtfR, mod.YtfI,
                                                    mod.FrVmBound, mod.ToVmBound, mod.FrVaBound, mod.ToVaBound)
            else
                # tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_br shmem=shmem_size polar_kernel_two_level_alternative(mod.n, mod.nline, mod.line_start, par.scale,
                #                                                 sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho,
                #                                                 shift_lines, mod.membuf, mod.YffR, mod.YffI, mod.YftR, mod.YftI,
                #                                                 mod.YttR, mod.YttI, mod.YtfR, mod.YtfI, mod.FrVmBound, mod.ToVmBound)
            end
            time_br += tgpu.time
            tgpu = CUDA.@timed @cuda threads=32 blocks=nblk_bus uc_bus_kernel_two_level_alternative(mod.baseMVA, mod.nbus, mod.t, mod.gen_start, mod.line_start,
                                                                        mod.FrStart, mod.FrIdx, mod.ToStart, mod.ToIdx, mod.GenStart,
                                                                        mod.GenIdx, mod.Pd, mod.Qd, sol.u_curr, sol.v_curr,
                                                                        sol.z_curr, sol.l_curr, sol.rho, mod.YshR, mod.YshI)
            time_bus += tgpu.time

            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_zv_kernel(mod.nvar, sol.u_curr, sol.v_curr, sol.z_curr, sol.l_curr, sol.rho, sol.lz, beta)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_l_kernel(mod.nvar, sol.l_curr, sol.z_curr, sol.lz, beta)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) compute_primal_residual_v_kernel(mod.nvar, sol.rp, sol.u_curr, sol.v_curr, sol.z_curr)
            @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.rd, sol.z_curr, sol.z_prev)
            CUDA.synchronize()

            CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) vector_difference(mod.nvar, sol.Ax_plus_By, sol.rp, sol.z_curr)

            mismatch = CUDA.norm(sol.Ax_plus_By)
            primres = CUDA.norm(sol.rp)
            dualres = CUDA.norm(sol.rd)
            z_curr_norm = CUDA.norm(sol.z_curr)
            eps_pri = sqrt_d / (2500*outer)

            if par.verbose > 0
                if inner == 1 || (inner % 50) == 0
                    @printf("%8s  %8s  %10s  %10s  %10s  %10s  %10s  %10s  %10s\n",
                            "Outer", "Inner", "PrimRes", "EpsPrimRes", "DualRes", "||z||",
                            "||Ax+By||", "OuterTol", "Beta")
                end

                @printf("%8d  %8d  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e  %10.3e\n",
                        outer, inner, primres, eps_pri, dualres, z_curr_norm, mismatch, OUTER_TOL, beta)
            end

            if primres <= eps_pri || dualres <= par.DUAL_TOL
                break
            end
        end # while inner

        if mismatch <= OUTER_TOL
            break
        end

        CUDA.@sync @cuda threads=64 blocks=(div(mod.nvar-1, 64)+1) update_lz_kernel(mod.nvar, par.MAX_MULTIPLIER, sol.z_curr, sol.lz, beta)

        if z_curr_norm > theta*z_prev_norm
            beta = min(c*beta, 1e24)
        end
    end # while outer
    end # @timed

    u_curr = zeros(mod.nvar)
    v_curr = zeros(mod.nvar)
    copyto!(u_curr, sol.u_curr)
    copyto!(v_curr, sol.v_curr)

    pg_err, qg_err = uc_check_generator_bounds(mod, u_curr)
    vm_err = uc_check_voltage_bounds_alternative(mod, v_curr)
    real_err, reactive_err = uc_check_power_balance_alternative(mod, u_curr, v_curr)
    rateA_nviols, rateA_maxviol = uc_check_linelimit_violation(data, v_curr, mod.t)

    sol.objval = sum(sum(data.generators[g].coeff[data.generators[g].n-2]*(mod.baseMVA*u_curr[mod.gen_start+2*mod.t*(g-1)+2*(tt-1)])^2 +
                    data.generators[g].coeff[data.generators[g].n-1]*(mod.baseMVA*u_curr[mod.gen_start+2*mod.t*(g-1)+2*(tt-1)]) +
                    data.generators[g].coeff[data.generators[g].n]
                         for g in 1:mod.ngen)
                    for tt=1:mod.t)::Float64
    sol.cumul_iters = cumul
    sol.overall_time = overall_time.time
    sol.status = (mismatch <= OUTER_TOL) ? :Solved : :IterLimit
    sol.max_viol_except_line = max(pg_err, qg_err, vm_err, real_err, reactive_err)
    sol.max_line_viol_rateA = rateA_maxviol

    if par.verbose > 0
        @printf(" ** Constraint violations \n")
        @printf("Real power generator bounds      = %.6e\n", pg_err)
        @printf("Reactive power generator bounds  = %.6e\n", qg_err)
        @printf("Voltage bounds                   = %.6e\n", vm_err)
        @printf("Real power balance               = %.6e\n", real_err)
        @printf("Reactive power balance           = %.6e\n", reactive_err)
        @printf("Number of line limit violations  = %d (%d)\n", rateA_nviols, mod.nline*mod.t)
        @printf("Maximum violation of line limit  = %.6e\n", rateA_maxviol)
        @printf("Maximum constraint violation     = %.6e\n", max(pg_err, qg_err, vm_err, real_err, reactive_err, rateA_maxviol))

        @printf(" ** Statistics\n")
        @printf("Objective value  . . . . . . . . . %12.6e\n", sol.objval)
        @printf("Outer iterations . . . . . . . . . %12d\n", outer)
        @printf("Cumulative iterations  . . . . . . %12d\n", cumul)
        @printf("Time per iteration . . . . . . . . %12.3f (secs/iter)\n", overall_time.time / cumul)
        @printf("Overall time . . . . . . . . . . . %12.3f (secs)\n", overall_time.time)
        @printf("Generator time . . . . . . . . . . %12.3f (secs)\n", time_gen)
        @printf("Branch time. . . . . . . . . . . . %12.3f (secs)\n", time_br)
        @printf("Bus time . . . . . . . . . . . . . %12.3f (secs)\n", time_bus)
        @printf("G+Br+B time. . . . . . . . . . . . %12.3f (secs)\n", time_gen + time_br + time_bus)
    end

    return
end
