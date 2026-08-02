
# Funzioni di simulazione per rettangolare vs CLEAR.

using DifferentialEquations: ODEProblem, Tsit5, solve
using Plots
using LaTeXStrings
using Plots.PlotMeasures

Plots.default(
    fontfamily = "Computer Modern",
    linewidth = 2,
    size = (1500, 1000),
    dpi = 300,
    guidefont = Plots.font(16),
    tickfont = Plots.font(13),
    legendfont = Plots.font(12),
    titlefont = Plots.font(18),
    left_margin = 14mm,
    bottom_margin = 12mm,
    right_margin = 8mm,
    top_margin = 8mm,
)

struct ResonatorParams
    kappa::Float64
    detuning::Float64
    alpha0::ComplexF64
    chi::Float64
end

ResonatorParams(kappa, detuning; alpha0=0.0 + 0.0im, chi=0.0) =
    ResonatorParams(float(kappa), float(detuning), ComplexF64(alpha0), float(chi))
    
function compare_rect_CLEAR(
    params::ResonatorParams;
    A,
    t_kick,
    t_readout,
    saveat_points=1500,
    reltol=1e-8,
    abstol=1e-10,
)
    kicks = clear_kick_amplitudes(
        kappa=params.kappa,
        detuning=params.detuning,
        chi=params.chi,
        A=A,
        t_kick=t_kick,
    )

    t_total = duration_CLEAR(t_kick=t_kick, t_readout=t_readout)
    t_rect_stop = 2 * t_kick + t_readout
    saveat = range(0.0, t_total, length=saveat_points)

    eps_rect(t) = epsilon_rect(t; A=A, t_start=0.0, t_stop=t_rect_stop)

    eps_clear(t) = epsilon_CLEAR(
        t;
        kappa=params.kappa,
        detuning=params.detuning,
        chi=params.chi,
        A=A,
        t_kick=t_kick,
        t_readout=t_readout,
        kicks=kicks,
    )

    

    return (
        epsilon_rect = eps_rect,
        epsilon_CLEAR = eps_clear,
        kicks = kicks,
        t_total = t_total,
        t_rect_stop = t_rect_stop,
        t_clear_ringdown_start = t_rect_stop,
        saveat = saveat,
    )
end



function plot_epsilon_rect(t_kick, t_readout, A; kwargs...)
    t_total = duration_CLEAR(
        t_kick=t_kick,
        t_readout=t_readout,
    )

    t_rect_stop = 2 * t_kick + t_readout

    eps_rect(t) = epsilon_rect(
        t;
        A=A,
        t_start=0.0,
        t_stop=t_rect_stop,
    )

    plt = Plots.plot(
        eps_rect;
        tspan=(0.0, t_total),
        xlims=(0.0, t_total),
        widen=false,
        label=L"\epsilon_{\mathrm{rect}}(t)",
        xlabel=L"t",
        ylabel=L"\epsilon(t)",
        kwargs...,
    )

    return plt
end

function plot_epsilon_clear(
    params::ResonatorParams,
    t_kick,
    t_readout,
    A;
    kwargs...
)
    t_total = duration_CLEAR(
        t_kick=t_kick,
        t_readout=t_readout,
    )

    kicks = clear_kick_amplitudes(
        kappa=params.kappa,
        detuning=params.detuning,
        chi=params.chi,
        A=A,
        t_kick=t_kick,
    )

    eps_clear(t) = epsilon_CLEAR(
        t;
        kappa=params.kappa,
        detuning=params.detuning,
        chi=params.chi,
        A=A,
        t_kick=t_kick,
        t_readout=t_readout,
        t_start=0.0,
        kicks=kicks,
    )

    t_plot = range(0.0, t_total, length=1000)
    epsilon_plot = real.(eps_clear.(t_plot))

    plt = Plots.plot(
        t_plot,
        epsilon_plot;
        label=L"\epsilon_{\mathrm{CLEAR}}(t)",
        xlabel=L"t\;(\mu\mathrm{s})",
        ylabel=L"\epsilon(t)",
        xlims=(0.0, t_total),
        widen=false,
        kwargs...,
    )

    return plt
end
    