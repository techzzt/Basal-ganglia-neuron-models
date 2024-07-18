# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)

#### Instantiating a LIF model network:

#### Parameters/Dimensions & Variable names:

Eqns | Code | Description
--- | --- | ---
$V(t)$ | `V` | Membrane potential
$V_r$ | `V_r` | Resting membrane potential
$V_t$ | `V_t` | Instantaneous "threshold" potential
$k$ | `k` | Constant ("$1/R$") 
$U(t)$ | `U` | Recovery variable
$a$ | `a` | Recovery time constant 
$b$ | `b` | Constant ("$1/R$")
$c$ | `V_reset` | Reset membrane potential
$d$ | `d` | Outward-minus-Inward currents activated during spike (affecting post-spike behavior)
$g_E(t)$ | `g_excit` | Excitatory conductance
$g_I(t)$ | `g_inhib` | Inhibitory conductance
$g_{gap}$ | `g_gap` | Gap junction conductance 
$V_E$ | `V_excit` | Equilibrium excitatory membrane potential 
$V_I$ | `V_inhib` | Equilibrium inhibitory membrane potential 
$V_{peak}$ | `V_peak` | Spike cutoff potential
$w_{gap}^{ji}$ | `ConnectionWeights_gap[j,i]` | Weight of gap junction connection between neurons $j$ and $i$ -- 
$I_{inp}$ | `I_inp` | External input current
$R_m$ | `R_membrane` | Membrane resitance
