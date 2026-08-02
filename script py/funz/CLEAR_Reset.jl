module CLEAR_Reset

include("impulsi_readout.jl")
include("simulazione.jl")

export ResonatorParams, epsilon_rect, steady_state_alpha, clear_kick_amplitudes, duration_CLEAR, epsilon_CLEAR, compare_rect_CLEAR, plot_epsilon_rect, plot_epsilon_clear  

end