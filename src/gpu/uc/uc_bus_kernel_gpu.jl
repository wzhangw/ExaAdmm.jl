function uc_bus_kernel_two_level_alternative(
    baseMVA::Float64, nbus::Int, t::Int, gen_start::Int, line_start::Int,
    FrStart::CuDeviceArray{Int,1}, FrIdx::CuDeviceArray{Int,1},
    ToStart::CuDeviceArray{Int,1}, ToIdx::CuDeviceArray{Int,1},
    GenStart::CuDeviceArray{Int,1}, GenIdx::CuDeviceArray{Int,1},
    Pd::CuDeviceArray{Float64,1}, Qd::CuDeviceArray{Float64,1},
    u::CuDeviceArray{Float64,1}, v::CuDeviceArray{Float64,1},
    z::CuDeviceArray{Float64,1}, l::CuDeviceArray{Float64,1}, rho::CuDeviceArray{Float64,1},
    YshR::CuDeviceArray{Float64,1}, YshI::CuDeviceArray{Float64,1}
)
    J = threadIdx().x + (blockDim().x * (blockIdx().x - 1))
    I = div(J, t, RoundUp)
    tidx = Base.mod(J, t) == 0 ? t : Base.mod(J, t)
    if J <= nbus*t
        common_wi = 0.0
        common_ti = 0.0
        inv_rhosum_pij_ji = 0.0
        inv_rhosum_qij_ji = 0.0
        rhosum_wi_ij_ji = 0.0
        rhosum_ti_ij_ji = 0.0

        @inbounds begin
            if FrStart[I] < FrStart[I+1]
                for k=FrStart[I]:FrStart[I+1]-1
                    pijt_idx = line_start + 8*(FrIdx[k]-1)*t + 8*(tidx-1)
                    common_wi += l[pijt_idx+4] + rho[pijt_idx+4]*(u[pijt_idx+4] + z[pijt_idx+4])
                    common_ti += l[pijt_idx+6] + rho[pijt_idx+6]*(u[pijt_idx+6] + z[pijt_idx+6])
                    inv_rhosum_pij_ji += 1.0 / rho[pijt_idx]
                    inv_rhosum_qij_ji += 1.0 / rho[pijt_idx+1]
                    rhosum_wi_ij_ji += rho[pijt_idx+4]
                    rhosum_ti_ij_ji += rho[pijt_idx+6]
                end
            end

            if ToStart[I] < ToStart[I+1]
                for k=ToStart[I]:ToStart[I+1]-1
                    pijt_idx = line_start + 8*(ToIdx[k]-1)*t + 8*(tidx-1)
                    common_wi += l[pijt_idx+5] + rho[pijt_idx+5]*(u[pijt_idx+5] + z[pijt_idx+5])
                    common_ti += l[pijt_idx+7] + rho[pijt_idx+7]*(u[pijt_idx+7] + z[pijt_idx+7])
                    inv_rhosum_pij_ji += 1.0 / rho[pijt_idx+2]
                    inv_rhosum_qij_ji += 1.0 / rho[pijt_idx+3]
                    rhosum_wi_ij_ji += rho[pijt_idx+5]
                    rhosum_ti_ij_ji += rho[pijt_idx+7]
                end
            end
        end

        common_wi /= rhosum_wi_ij_ji

        rhs1 = 0.0
        rhs2 = 0.0
        inv_rhosum_pg = 0.0
        inv_rhosum_qg = 0.0

        @inbounds begin
            if GenStart[I] < GenStart[I+1]
                for g=GenStart[I]:GenStart[I+1]-1
                    pgt_idx = gen_start + 2*(GenIdx[g]-1)*t + 2*(tidx-1)
                    rhs1 += (u[pgt_idx] + z[pgt_idx]) + (l[pgt_idx]/rho[pgt_idx])
                    rhs2 += (u[pgt_idx+1] + z[pgt_idx+1]) + (l[pgt_idx+1]/rho[pgt_idx+1])
                    inv_rhosum_pg += 1.0 / rho[pgt_idx]
                    inv_rhosum_qg += 1.0 / rho[pgt_idx+1]
                end
            end

            rhs1 -= (Pd[I] / baseMVA)
            rhs2 -= (Qd[I] / baseMVA)

            if FrStart[I] < FrStart[I+1]
                for k=FrStart[I]:FrStart[I+1]-1
                    pijt_idx = line_start + 8*(FrIdx[k]-1)
                    rhs1 -= (u[pijt_idx] + z[pijt_idx]) + (l[pijt_idx]/rho[pijt_idx])
                    rhs2 -= (u[pijt_idx+1] + z[pijt_idx+1]) + (l[pijt_idx+1]/rho[pijt_idx+1])
                end
            end

            if ToStart[I] < ToStart[I+1]
                for k=ToStart[I]:ToStart[I+1]-1
                    pijt_idx = line_start + 8*(ToIdx[k]-1)
                    rhs1 -= (u[pijt_idx+2] + z[pijt_idx+2]) + (l[pijt_idx+2]/rho[pijt_idx+2])
                    rhs2 -= (u[pijt_idx+3] + z[pijt_idx+3]) + (l[pijt_idx+3]/rho[pijt_idx+3])
                end
            end

            rhs1 -= YshR[I]*common_wi
            rhs2 += YshI[I]*common_wi

            A11 = (inv_rhosum_pg + inv_rhosum_pij_ji) + (YshR[I]^2 / rhosum_wi_ij_ji)
            A12 = -YshR[I]*(YshI[I] / rhosum_wi_ij_ji)
            A21 = A12
            A22 = (inv_rhosum_qg + inv_rhosum_qij_ji) + (YshI[I]^2 / rhosum_wi_ij_ji)
            mu2 = (rhs2 - (A21/A11)*rhs1) / (A22 - (A21/A11)*A12)
            mu1 = (rhs1 - A12*mu2) / A11
            #mu = A \ [rhs1 ; rhs2]
            wi = common_wi + ( (YshR[I]*mu1 - YshI[I]*mu2) / rhosum_wi_ij_ji )
            ti = common_ti / rhosum_ti_ij_ji

            for k=GenStart[I]:GenStart[I+1]-1
                pgt_idx = gen_start + 2*(GenIdx[k]-1)*t + 2*(tidx-1)
                v[pgt_idx] = (u[pgt_idx] + z[pgt_idx]) + (l[pgt_idx] - mu1) / rho[pgt_idx]
                v[pgt_idx+1] = (u[pgt_idx+1] + z[pgt_idx+1]) + (l[pgt_idx+1] - mu2) / rho[pgt_idx+1]
            end
            for j=FrStart[I]:FrStart[I+1]-1
                pijt_idx = line_start + 8*(FrIdx[j]-1)*t + 8*(tidx-1)
                v[pijt_idx] = (u[pijt_idx] + z[pijt_idx]) + (l[pijt_idx] + mu1) / rho[pijt_idx]
                v[pijt_idx+1] = (u[pijt_idx+1] + z[pijt_idx+1]) + (l[pijt_idx+1] + mu2) / rho[pijt_idx+1]
                v[pijt_idx+4] = wi
                v[pijt_idx+6] = ti
            end
            for j=ToStart[I]:ToStart[I+1]-1
                pijt_idx = line_start + 8*(ToIdx[j]-1)*t + 8*(tidx-1)
                v[pijt_idx+2] = (u[pijt_idx+2] + z[pijt_idx+2]) + (l[pijt_idx+2] + mu1) / rho[pijt_idx+2]
                v[pijt_idx+3] = (u[pijt_idx+3] + z[pijt_idx+3]) + (l[pijt_idx+3] + mu2) / rho[pijt_idx+3]
                v[pijt_idx+5] = wi
                v[pijt_idx+7] = ti
            end
        end
    end

    return
end
