function uc_check_generator_bounds(model::UCModel, xbar::Vector{Float64})
    ngen = model.ngen
    gen_start = model.gen_start
    t = model.t

    pgmax = model.pgmax_curr; pgmin = model.pgmin_curr
    qgmax = model.qgmax; qgmin = model.qgmin

    max_viol_real = 0.0
    max_viol_reactive = 0.0

    for g=1:ngen
        for tt=1:t
            pidx = gen_start + 2*t*(g-1) + 2*(tt-1)
            qidx = gen_start + 2*t*(g-1) + 2*(tt-1) + 1

            real_err = max(max(0.0, xbar[pidx] - pgmax[g]), max(0.0, pgmin[g] - xbar[pidx]))
            reactive_err = max(max(0.0, xbar[qidx] - qgmax[g]), max(0.0, qgmin[g] - xbar[qidx]))

            max_viol_real = (max_viol_real < real_err) ? real_err : max_viol_real
            max_viol_reactive = (max_viol_reactive < reactive_err) ? reactive_err : max_viol_reactive
        end
    end

    return max_viol_real, max_viol_reactive
end

function uc_check_voltage_bounds_alternative(model::UCModel, v::Vector{Float64})
    max_viol = 0.0
    t = model.t

    for b=1:model.nbus
        for tt=1:t
            if model.FrStart[b] < model.FrStart[b+1]
                l = model.FrIdx[model.FrStart[b]]
                wi = v[model.line_start + 8*t*(l-1) + 8*(tt-1) + 4]
            elseif model.ToStart[b] < model.ToStart[b+1]
                l = model.ToIdx[model.ToStart[b]]
                wi = v[model.line_start + 8*t*(l-1) + 8*(tt-1) + 5]
            else
                println("No lines connected to bus ", b)
            end

            err = max(max(0.0, wi - model.Vmax[b]^2), max(0.0, model.Vmin[b]^2 - wi))
            max_viol = (max_viol < err) ? err : max_viol
        end
    end

    return max_viol
end

function uc_check_power_balance_alternative(model::UCModel, u::Vector{Float64}, v::Vector{Float64})
    baseMVA = model.baseMVA
    nbus = model.nbus
    gen_start, line_start, YshR, YshI = model.gen_start, model.line_start, model.YshR, model.YshI
    t = model.t

    max_viol_real = 0.0
    max_viol_reactive = 0.0
    for b=1:nbus
        for tt=1:t
            real_err = 0.0
            reactive_err = 0.0
            for k=model.GenStart[b]:model.GenStart[b+1]-1
                g = model.GenIdx[k]
                real_err += u[gen_start + 2*t*(g-1) + 2*(tt-1)]
                reactive_err += u[gen_start+2*t*(g-1)+2*(tt-1)+1]
            end

            real_err -= (model.Pd[b] / baseMVA)
            reactive_err -= (model.Qd[b] / baseMVA)

            wi = 0
            for k=model.FrStart[b]:model.FrStart[b+1]-1
                l = model.FrIdx[k]
                real_err -= v[line_start + 8*t*(l-1) + 8*(tt-1)]
                reactive_err -= v[line_start + 8*t*(l-1) + 8*(tt-1) + 1]
                wi = v[line_start + 8*t*(l-1) + 8*(tt-1) + 4]
            end

            for k=model.ToStart[b]:model.ToStart[b+1]-1
                l = model.ToIdx[k]
                real_err -= v[line_start + 8*t*(l-1) + 8*(tt-1) + 2]
                reactive_err -= v[line_start + 8*t*(l-1) + 8*(tt-1) + 3]
                wi = v[line_start + 8*t*(l-1) + 8*(tt-1) + 5]
            end

            real_err -= YshR[b] * wi
            reactive_err += YshI[b] * wi

            max_viol_real = (max_viol_real < abs(real_err)) ? abs(real_err) : max_viol_real
            max_viol_reactive = (max_viol_reactive < abs(reactive_err)) ? abs(reactive_err) : max_viol_reactive
        end
    end

    return max_viol_real, max_viol_reactive
end

function uc_check_linelimit_violation(data::UCOPFData, v::Vector{Float64}, t::Int)
    lines = data.lines
    nline = length(data.lines)
    line_start = 2*length(data.generators)*t + 1

    rateA_nviols = 0
    rateA_maxviol = 0.0

    for l=1:nline
        for tt=1:t
            pijt_idx = line_start + 8*t*(l-1) + 8*(tt-1)
            ij_val = sqrt(v[pijt_idx]^2 + v[pijt_idx+1]^2)
            ji_val = sqrt(v[pijt_idx+2]^2 + v[pijt_idx+3]^2)

            limit = lines[l].rateA / data.baseMVA
            if limit > 0
                if ij_val > limit || ji_val > limit
                    rateA_nviols += 1
                    rateA_maxviol = max(rateA_maxviol, max(ij_val - limit, ji_val - limit))
                end
            end
        end
    end
    rateA_maxviol = sqrt(rateA_maxviol)

    return rateA_nviols, rateA_maxviol
end
