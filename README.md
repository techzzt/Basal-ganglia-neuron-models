# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)

#### Instantiating a LIF model network:

#### Parameters/Dimensions & Variable names:

Eqns | Code | Description
--- | --- | ---
$V$ | `v` | Membrane potential 
$V_r$ | `vr` | Resting membrane potential
$V_t$ | `vt` | Instantaneous "threshold" potential
$V_{peak}$ | `vpeak` | Spike cutoff potential
$k$ | `k` | Constant ("$1/R$") 
$U(t)$ | `U` | Recovery variable
$a$ | `a` | Recovery time constant 
$b$ | `b` | Constant ("$1/R$")
$c$ | `V_reset` | Reset membrane potential
$d$ | `d` | Outward-minus-Inward currents activated during spike (affecting post-spike behavior)
$C$ | `C` | Conductance
$I$ | `I` | input current
