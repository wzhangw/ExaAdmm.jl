function uc_generator_kernel_two_level(
    baseMVA::Float64, ngen::Int, t::Int, gen_start::Int,
    u::CuDeviceArray{Float64,1}, x::CuDeviceArray{Float64,1}, z::CuDeviceArray{Float64,1},
    l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    pgmin::CuDeviceArray{Float64,1}, pgmax::CuDeviceArray{Float64,1},
    qgmin::CuDeviceArray{Float64,1}, qgmax::CuDeviceArray{Float64,1},
    c2::CuDeviceArray{Float64,1}, c1::CuDeviceArray{Float64,1}
)
    I = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    gen_idx = div(I, t, RoundUp)
    # tidx = Base.mod(J, t) == 0 ? t : Base.mod(J, t)
    if (I <= ngen*t)
        pg_idx = gen_start + 2*(I-1)
        qg_idx = gen_start + 2*(I-1) + 1
        u[pg_idx] = max(pgmin[gen_idx],
                        min(pgmax[gen_idx],
                            (-(c1[gen_idx]*baseMVA + l[pg_idx] + rho[pg_idx]*(-x[pg_idx] + z[pg_idx]))) / (2*c2[gen_idx]*(baseMVA^2) + rho[pg_idx])))
        u[qg_idx] = max(qgmin[gen_idx],
                        min(qgmax[gen_idx],
                            (-(l[qg_idx] + rho[qg_idx]*(-x[qg_idx] + z[qg_idx]))) / rho[qg_idx]))
    end

    return
end

function uc_generator_kernel_two_level(
    model::UCModel{Float64,CuArray{Float64,1},CuArray{Int,1}},
    baseMVA::Float64, u::CuArray{Float64,1}, xbar::CuArray{Float64,1},
    zu::CuArray{Float64,1}, lu::CuArray{Float64,1}, rho_u::CuArray{Float64,1}
)
    nblk = div(model.ngen*model.t, 32, RoundUp)
    tgpu = CUDA.@timed @cuda threads=32 blocks=nblk uc_generator_kernel_two_level(baseMVA, model.ngen, model.t, model.gen_start,
                u, xbar, zu, lu, rho_u, model.pgmin_curr, model.pgmax_curr, model.qgmin, model.qgmax, model.c2, model.c1)
    return tgpu
end