# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)


### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters | Condition
--- | --- | --- | ---
LIF | $$\frac{dv}{dt} = \frac{-g_L \cdot 1 \, \text{pF/ms/mV} \cdot (v - E_L) + I}{C}$$ | g<sub>L</sub>, E<sub>L</sub>, d, vr, th, C | 
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$ <br> <br> $$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | a, b, c, C, d, k, vr, th, vpeak, u, I | $$v>vpeak, v = c, u = u + d$$ 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$ <br> <br> $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g<sub>L</sub>, E<sub>L</sub>, Delta<sub>T</sub>, vr, vt, th, tau<sub>w</sub>, a, d, C | v>t<sup>f</sup>  v = c, u = u + d
Izhikevich |$$\frac{dv}{dt} = \left(0.04 \, \frac{1}{\text{ms} \cdot \text{mV}}\right)v^2 + \left(5 \, \frac{1}{\text{ms}}\right)v + 140 \,\frac{\text{mV}}{\text{ms}} - u + I \quad$$ <br> <br> $$\frac{du}{dt} = a \left(bv - u\right) \quad$$ | a, b, d, vr |

- Transient Response of Basal Ganglia Network in Healthy and Low-Dopamine State


## ARGUMENTS
neuron models should be put under ```./bg_insilico```

- models/model{}.py: define quadratic integrate and fire model (AdEx, LIF, QIF, HH, Izh)
- params: parameter list with json format
- simulation_runner.py: generate neuron model, spike mornitoring
- result.py: plotting 
- test.ipynb: define params, simulation


#### Neuron Model Features
Neuron type | Brain region | Description | References | Alternative terms | Neuron transmitter | Json
--- | --- | --- | --- | --- | ---| ---
MSN D1 | Striatum | LIF model [2] | | | GABA | 
MSN D2 | Striatum | LIF model, set constant input as 0 [2] | | | GABA |
FSN | Striatum | | | FSI [1], FS[3] | GABA |
ChiN | Striatum |  | | | ACh |
prototypic | GPe | [2] refer to [1] | | GPe-T1 | |
Arkypallidal | GPe | [2] refer to [1] | | GPe-TA | |
PV+ | STN | not seperate STN type [1] | | | | 
PV- | STN |  | | | |
PV+ | GPi |  | | | |
SST+ | GPi |  | | | |
PV+ | SNr | not seperate SNr type [1] <br> spike cut off refer to [1], and set I arbitarily [2] | | | |
SST+ | SNr |  | | | |
GABAergic | SNc |  | | | |
Dopaminergic | SNc |  | | | |

https://www.frontiersin.org/journals/neural-circuits/articles/10.3389/fncir.2018.00003/full
synapse connection code reference: https://github.com/Hjorthmedh/Snudda/blob/master/snudda/data/input_config/external-input-dSTR-scaled-v2.json

#### Population Value 
Neuron type | [References 1](https://www.eneuro.org/content/3/6/ENEURO.0156-16.2016.short) | References 2 | [Reference 3](https://www.eneuro.org/content/9/2/ENEURO.0376-21.2022.abstract) | [Reference 4](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010645)
--- | --- | --- | --- | --- 
Network | 80,000 |  | 6,539 |
MSN D1 | 37,971 | 2,000 | 2,000 | 6,000
MSN D2 | 37,971 | 2,000 | 2,000 | 6,000
FSN | 1,599 | 80 | 80 | 420
ChiN | - | - | - | -
prototypic (GPe) | 988 | - | 988 | 780
Arkypallidal (GPe) | 329 | - | 329 | 264
PV+ (STN) | **388** | - | **388** | **408**
PV- (STN) | - | - | - | -
PV+ (GPi) | - | - | - | -
SST+ (GPi) | - | - | - | - 
PV+ (SNr) | **754** | - | **754** | -
SST+ (SNr) | - | - | - | -
GABAergic (SNc) | - | - | - | -
Dopaminergic (SNc) | - | - | - | -

#### References (parameters)
[1] Lindahl, Mikael, and Jeanette Hellgren Kotaleski. "Untangling basal ganglia network dynamics and function: Role of dopamine depletion and inhibition investigated in a spiking network model." eneuro 3.6 (2016).

[2] Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

[3] Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.

#### References (others)
[1] Giordano, Nadia, et al. "Fast-spiking interneurons of the premotor cortex contribute to initiation and execution of spontaneous actions." Journal of Neuroscience 43.23 (2023): 4234-4250.

[2] Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

[3] Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


- Kim et al. (2024): Kim, Sang-Yoon, and Woochang Lim. "Quantifying harmony between direct and indirect pathways in the basal ganglia: healthy and Parkinsonian states." Cognitive Neurodynamics (2024): 1-21.
- Sheng, M. J., Lu, D., Shen, Z. M., and Poo, M. M. (2019). Emergence of stable striatal D1R and D2R neuronal ensembles with distinct firing sequence during motor learning. Proc. Natl. Acad. Sci. U. S. A. 116, 11038–11047. doi: 10.1073/pnas.1901712116
- Surmeier, Dalton James, et al. "Rethinking the network determinants of motor disability in Parkinson’s disease." Frontiers in Synaptic Neuroscience 15 (2023): 1186484.
- Kumaravelu, Karthik, David T. Brocker, and Warren M. Grill. "A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease." Journal of computational neuroscience 40 (2016): 207-229.
- McCarthy, M. M., et al. "Striatal origin of the pathologic beta oscillations in Parkinson's disease." Proceedings of the national academy of sciences 108.28 (2011): 11620-11625.
- Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.
