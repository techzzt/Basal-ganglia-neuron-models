# Basal-ganglia-neuron-models

**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)
![neurotransmitter](https://github.com/user-attachments/assets/4bd7e1a5-2c19-457f-a619-24c511868422)


### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters | Condition
--- | --- | --- | ---
LIF | $$\frac{dv}{dt} = \frac{-g_L \cdot 1 \, \text{pF/ms/mV} \cdot (v - E_L) + I}{C}$$ | g<sub>L</sub>, E<sub>L</sub>, d, vr, th, C | 
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$ <br> <br> $$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | a, b, c, C, d, k, vr, th, vpeak, u, I | $$v>vpeak, v = c, u = u + d$$ 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$ <br> <br> $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g<sub>L</sub>, E<sub>L</sub>, Delta<sub>T</sub>, vr, vt, th, tau<sub>w</sub>, a, d, C | v>t<sup>f</sup>, v = c, u = u + d
Izhikevich |$$\frac{dv}{dt} = \left(0.04 \, \frac{1}{\text{ms} \cdot \text{mV}}\right)v^2 + \left(5 \, \frac{1}{\text{ms}}\right)v + 140 \,\frac{\text{mV}}{\text{ms}} - u + I \quad$$ <br> <br> $$\frac{du}{dt} = a \left(bv - u\right) \quad$$ | a, b, d, vr |


## ARGUMENTS
neuron models should be put under ```./bg_insilico```

- models/model{}.py: define quadratic integrate and fire model (AdEx, LIF, QIF, HH, Izh)
- params: parameter list with json format
- simulation_runner.py: generate neuron model, spike mornitoring
- simulation_runner.py: generate neuron model, spike mornitoring with various I 
- test.ipynb: define params, simulation


#### Neuron Model Features
Neuron type | Brain region | Parameter Description | References | Alternative terms | Neuron transmitter | Json
--- | --- | --- | --- | --- | ---| ---
MSN D1 | Striatum | LIF model [2] | | STN, dSPN [4] | GABA | 
MSN D2 | Striatum | LIF model, set constant input as 0 [2] | | iSPN | GABA |
FSN | Striatum | | | FSI [1], FS[3] | GABA |
ChiN | Striatum |  | | | ACh |
prototypic | GPe | [2], [3] refer to [1] | consider ChAT in GPe [5] | GPe-T1 | GABA |
Arkypallidal | GPe | [2], [3] refer to [1] | | GPe-TA | GABA |
PV+ | STN | not seperate STN type [1] | | | glu | 
PV- | STN |  | | | glu |
PV+ | GPi |  | | | GABA |
SST+ | GPi |  | | | GABA |
PV+ | SNr | not seperate SNr type [1] <br> spike cut off refer to [1], and set I arbitarily [2] | | | GABA [6] |
SST+ | SNr |  | | | GABA |
GABAergic | SNc |  | | | |
Dopaminergic | SNc |  | | | |

#### Population Value 
Neuron type | [References 1](https://www.eneuro.org/content/3/6/ENEURO.0156-16.2016.short) | References 2 | [Reference 3](https://www.eneuro.org/content/9/2/ENEURO.0376-21.2022.abstract) | [Reference 4](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010645) | [Reference 5](https://www.biorxiv.org/content/10.1101/2023.03.07.531640v1)
--- | --- | --- | --- | --- | ---
Network | 80,000 |  | 6,539 | - | 
MSN D1 | 37,971 | 2,000 | 2,000 | 6,000 | -
MSN D2 | 37,971 | 2,000 | 2,000 | 6,000 | 13,300,000 (0.475 * 2.8 * 10<sup>6</sup>)
FSN | 1,599 | 80 | 80 | 420 | 11,500,000 (0.2 * 2.8 * 10<sup>6</sup>)
ChiN | - | - | - | - | -
prototypic (GPe) | 988 | - | 988 | 780 | 322,000 (0.7 * 4.6 * 10<sup>4</sup>)
Arkypallidal (GPe) | 329 | - | 329 | 264 | 115,000 (0.25 * 4.6 * 10<sup>4</sup>)
**STN total** | **388** | - | **388** | **408** | **13,560**
PV+ (STN) | **155 (~40%)** | - | - | - | -
PV- (STN) | **233 (~60%)** | - | - | - | -
PV+ (GPi) | - | - | - | - | -
SST+ (GPi) | - | - | - | - | -
PV+ (SNr) | **754** | - | **754** | - | -
SST+ (SNr) | - | - | - | - | -
GABAergic (SNc) | - | - | - | - | -
Dopaminergic (SNc) | - | - | - | - | -

**Note on STN PV+/PV- subdivision**: The subdivision of STN into PV+ (~40%) and PV- (~60%) populations is based on experimental evidence showing that approximately 30-40% of STN neurons express parvalbumin [7,8]. This proportion is consistent with immunohistochemical studies in rodent models.

#### References (parameters)
[1] Lindahl, Mikael, and Jeanette Hellgren Kotaleski. "Untangling basal ganglia network dynamics and function: Role of dopamine depletion and inhibition investigated in a spiking network model." eneuro 3.6 (2016).

[2] Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

[3] Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.

#### References (others)
[1] Giordano, Nadia, et al. "Fast-spiking interneurons of the premotor cortex contribute to initiation and execution of spontaneous actions." Journal of Neuroscience 43.23 (2023): 4234-4250.

[2] Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

[3] Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.

[4] Fieblinger, Tim. "Striatal control of movement: a role for new neuronal (Sub-) populations?." Frontiers in human neuroscience 15 (2021): 697284.

[5] Dong, Jie, et al. "Connectivity and functionality of the globus pallidus externa under normal conditions and Parkinson's disease." Frontiers in neural circuits 15 (2021): 645287.

[6] The role of neurotransmitter systems in mediating deep brain stimulation effects in Parkinson's disease

[7] Lévesque, Martin, and André Parent. "The striatofugal fiber system in primates: a reevaluation of its organization based on single‐axon tracing studies." Proceedings of the National Academy of Sciences 102.33 (2005): 11888-11893.

[8] Cooper, Andrew J., and Jeffery R. Wickens. "Measurement of the membrane time constant and yield of GABA-mediated inhibition in the subthalamic nucleus." Journal of neurophysiology 103.2 (2010): 1049-1059.

- Kim et al. (2024): Kim, Sang-Yoon, and Woochang Lim. "Quantifying harmony between direct and indirect pathways in the basal ganglia: healthy and Parkinsonian states." Cognitive Neurodynamics (2024): 1-21.

