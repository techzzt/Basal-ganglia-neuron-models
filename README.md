# Basal-ganglia-neuron-models

**Network Structure**
<img width="1151" alt="prototype_structure" src="https://github.com/user-attachments/assets/336e1cd1-16f9-4d9a-b520-6db1933aeb92">


### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters | Condition
--- | --- | --- | ---
LIF | $$\frac{dv}{dt} = \frac{-g_L \cdot 1 \, \text{pF/ms/mV} \cdot (v - E_L) + I}{C}$$ | g<sub>L</sub>, E<sub>L</sub>, d, vr, th, C | 
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$ <br> <br> $$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | a, b, c, C, d, k, vr, th, vpeak, u, I | $$v>vpeak, v = c, u = u + d$$ 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$ <br> <br> $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g<sub>L</sub>, E<sub>L</sub>, Delta<sub>T</sub>, vr, vt, th, tau<sub>w</sub>, a, d, C | v>t<sup>f</sup>, v = c, u = u + d
Izhikevich |$$\frac{dv}{dt} = \left(0.04 \, \frac{1}{\text{ms} \cdot \text{mV}}\right)v^2 + \left(5 \, \frac{1}{\text{ms}}\right)v + 140 \,\frac{\text{mV}}{\text{ms}} - u + I \quad$$ <br> <br> $$\frac{du}{dt} = a \left(bv - u\right) \quad$$ | a, b, d, vr |


## ARGUMENTS
neuron models should be put under ```./bg_insilico```

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
PV+ (STN) | **388** | - | **388** | **408** | **13,560**
PV- (STN) | - | - | - | - | -
PV+ (GPi) | - | - | - | - | -
SST+ (GPi) | - | - | - | - | -
PV+ (SNr) | **754** | - | **754** | - | -
SST+ (SNr) | - | - | - | - | -
GABAergic (SNc) | - | - | - | - | -
Dopaminergic (SNc) | - | - | - | - | -

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

[6] The role of neurotransmitter systems in mediating deep brain stimulation effects in Parkinson’s disease

--- 

McGregor, Matthew M., and Alexandra B. Nelson. "Circuit mechanisms of Parkinson’s disease." Neuron 101.6 (2019): 1042-1056.

Foster, Nicholas N., et al. "The mouse cortico–basal ganglia–thalamic network." Nature 598.7879 (2021): 188-194.

