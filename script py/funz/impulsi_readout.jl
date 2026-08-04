
# Simiclassical readout pulses: rectangular and CLEAR.


#Rectangular pulse. Returns A for t_start <= t < t_stop, and zero outside the interval.

function epsilon_rect(t; A=1.0 + 0im, t_start=0.0, t_stop=1.0)
    t_stop >= t_start || error("t_stop must be >= t_start for epsilon_rect.")

    if t_start <= t < t_stop
        return A
    else
        return zero(A)
    end
end


#=Steady-state coherent amplitude of the resonator for constant drive `A`
α_ss = -i A / (κ/2 + iΔ).=#

function steady_state_alpha(kappa, detuning, A)
    kappa > 0 || error("kappa must be > 0 for steady_state_alpha.")
    lambda = kappa / 2 + im * detuning
    return -im * A / lambda
end

# Analytical propagation with constant drive: initial value alpha0, drive epsilon for time tau.

function _propagate_constant_drive(alpha0, epsilon, kappa, detuning, tau)
    tau >= 0 || error("tau must be >= 0.")
    lambda = kappa / 2 + im * detuning
    c = exp(-lambda * tau)
    return c * alpha0 - im * epsilon * (1 - c) / lambda
end

# Calculate two constant amplitudes epsilon1, epsilon2 such that, after two segments of equal duration tau, two trajectories with different detunings reach the targets.

function _two_kick_amplitudes(kappa, detunings, alpha_initials, alpha_targets, tau)
    kappa > 0 || error("kappa must be > 0.")
    tau > 0 || error("tau must be > 0.")
    length(detunings) == 2 || error("detunings must have length 2.")
    length(alpha_initials) == 2 || error("alpha_initials must have length 2.")
    length(alpha_targets) == 2 || error("alpha_targets must have length 2.")

    M = Matrix{ComplexF64}(undef, 2, 2)
    b = Vector{ComplexF64}(undef, 2)

    for j in 1:2
        lambda = kappa / 2 + im * detunings[j]
        c = exp(-lambda * tau)

        # Dopo due segmenti:
        # α_f = c^2 α_i - i(1-c)c/λ * ε1 - i(1-c)/λ * ε2
        M[j, 1] = -im * (1 - c) * c / lambda
        M[j, 2] = -im * (1 - c) / lambda
        b[j] = alpha_targets[j] - c^2 * alpha_initials[j]
    end

    eps = M \ b
    return eps[1], eps[2]
end


#= Calculate the four amplitudes of the CLEAR kicks in the linear resonator model.
The readout drive has amplitude `A`. The two trajectories conditioned on the ground/excited state are described by the effective detunings

Δ_g = detuning - chi
Δ_e = detuning + chi =#


function clear_kick_amplitudes(; kappa, detuning=0.0, chi, A=1.0 + 0im, t_kick)
    kappa > 0 || error("kappa must be > 0 for clear_kick_amplitudes.")
    t_kick > 0 || error("t_kick must be > 0 for clear_kick_amplitudes.")

    detuning_g = detuning - chi
    detuning_e = detuning + chi
    detunings = ComplexF64[detuning_g, detuning_e]

    alpha_ss_g = steady_state_alpha(kappa, detuning_g, A)
    alpha_ss_e = steady_state_alpha(kappa, detuning_e, A)

    up1, up2 = _two_kick_amplitudes(
        kappa,
        detunings,
        ComplexF64[0.0 + 0.0im, 0.0 + 0.0im],
        ComplexF64[alpha_ss_g, alpha_ss_e],
        t_kick,
    )

    down1, down2 = _two_kick_amplitudes(
        kappa,
        detunings,
        ComplexF64[alpha_ss_g, alpha_ss_e],
        ComplexF64[0.0 + 0.0im, 0.0 + 0.0im],
        t_kick,
    )

    return (
        ringup1 = up1,
        ringup2 = up2,
        readout = ComplexF64(A),
        ringdown1 = down1,
        ringdown2 = down2,
    )
end


#Total duration of the CLEAR pulse:

function duration_CLEAR(; t_kick, t_readout)
    t_kick > 0 || error("t_kick must be > 0 for duration_CLEAR.")
    t_readout >= 0 || error("t_readout must be >= 0 for duration_CLEAR.")
    return 4 * t_kick + t_readout
end


#= CLEAR pulse with five segments
The amplitudes of the kicks are calculated from the linear resonator model if `kicks=nothing`. To avoid recalculating them at each temporal evaluation, it is convenient to calculate them once with `clear_kick_amplitudes` and pass them here.=#

function epsilon_CLEAR(
    t;
    kappa,
    detuning=0.0,
    chi,
    A,
    t_kick,
    t_readout,
    t_start=0.0,
    kicks=nothing,
)
    t_kick > 0 || error("t_kick must be > 0 for epsilon_CLEAR.")
    t_readout >= 0 || error("t_readout must be >= 0 for epsilon_CLEAR.")

    local clear_kicks
    if isnothing(kicks)
        clear_kicks = clear_kick_amplitudes(
            kappa=kappa,
            detuning=detuning,
            chi=chi,
            A=A,
            t_kick=t_kick,
        )
    else
        clear_kicks = kicks
    end

    tt = t - t_start

    t1 = t_kick
    t2 = 2 * t_kick
    t3 = 2 * t_kick + t_readout
    t4 = 3 * t_kick + t_readout
    t5 = 4 * t_kick + t_readout

    if 0 <= tt < t1
        return clear_kicks.ringup1
    elseif t1 <= tt < t2
        return clear_kicks.ringup2
    elseif t2 <= tt < t3
        return clear_kicks.readout
    elseif t3 <= tt < t4
        return clear_kicks.ringdown1
    elseif t4 <= tt < t5
        return clear_kicks.ringdown2
    else
        return zero(ComplexF64(A))
    end
end
