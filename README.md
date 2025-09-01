# Basal-ganglia-neuron-models

**Network Structure**
<img width="1151" alt="prototype_structure" src="https://github.com/user-attachments/assets/336e1cd1-16f9-4d9a-b520-6db1933aeb92">


### Equation/Parameters & Variable names:

Eqns | eqs | Condition
--- | --- | ---
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$ <br> <br> $$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | $$v>vpeak, v = c, u = u + d$$ 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$ <br> <br> $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | v>t<sup>f</sup>, v = c, u = u + d

<img width="905" height="376" alt="image" src="https://github.com/user-attachments/assets/0bb82e8c-af4c-416b-ac2c-da2c7575f90d" />


## Arguments
neuron models should be put under ```./bg_insilico```

| Neuron Type | Brain Region | Model Type | Description | Alternative Terms | Transmitter | Implementation Status |
|-----------|--------------|-----------------|------------------|-----------------|---------|-----------------|
| MSN D1 | Striatum | QIF | Direct pathway, D1 receptors | dSPN, Direct SPN | GABA | Implemented | 
| MSN D2 | Striatum | QIF | Indirect pathway, D2 receptors | iSPN, Indirect SPN | GABA | Implemented | 
| FSN | Striatum | QIF and Type-2 | Fast-spiking, feedforward inhibition | FSI, FS | GABA | Implemented |
| ChIN | Striatum | HH | Tonically active | TANs | ACh | Planned | 
| Prototypic | GPe | AdEx | High firing rate | GPe-TI, Proto | GABA | Implemented |
| Arkypallidal | GPe | AdEx |  | GPe-TA, Arky | GABA | Implemented | 
| PV+ | STN | AdEx | High frequency bursting | paravalbumin PV+ | Glutamate | Planned | 
| PV- | STN | AdEx |  | Tonic bursiting | Glutamate | Implemented | 
| PV+ | GPi | Izhikevich | High-frequency tonic firing |  | GABA |  | 
| SST+ | GPi | Izhikevich | Lower baseline firing |  | GABA |  |
| PV+ | SNr | AdEx | High-frequency firing | SNr-PV+ | GABA | Implemented | 
| SST+ | SNr | AdEx | Heterogeneous firing, motor control |  | GABA |  | 
| GABAergic | SNc | LIF | Local inhibition |  | GABA |  | 
| Dopaminergic | SNc | HH | Phasic bursting | SNc-DA, DA neuron | DA + GABA |  |  |

#### Population Value 
Neuron type | [References 1](https://www.eneuro.org/content/3/6/ENEURO.0156-16.2016.short) | [Reference 2](https://www.eneuro.org/content/9/2/ENEURO.0376-21.2022.abstract) | [Reference 3](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1010645) | [Reference 4](https://www.biorxiv.org/content/10.1101/2023.03.07.531640v1) | [Reference 5](https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0189109&type=printable) | [Reference 6](https://www.biorxiv.org/content/10.1101/2023.09.05.556301v3.full.pdf)
--- | ---  | --- | --- | --- | --- | ---
Network | 80,000 | 6,539 | - | - | - 
MSN D1 | 37,971 | 2,000 | 6,000 | - | 2,762,100 | 75
MSN D2 | 37,971 | 2,000 | 6,000 | 13,300,000 | 2,762,100 | 75
FSN | 1,599 | 80 | 420 | 11,500,000 | 279,000 | 75
ChiN | - | - | - | - | - 
prototypic (GPe) | 988 | 988 | 780 | 322,000 | 46,000 | 190
Arkypallidal (GPe) | 329 | 329 | 264 | 115,000 | 46,000 | 560
PV+ (STN) | **388** | **388** | **408** | **13,560** | 13,600 | 750
PV- (STN) | - | - | - | - | - | - 
PV+ (GPi) | - | - | - | - | - | 75
SST+ (GPi) | - | - | - | - | - | -
PV+ (SNr) | **754** | **754** | - | - | 26,300 | -
SST+ (SNr) | - | - | - | - | 26,300 | -
GABAergic (SNc) | - | - | - | - | - | -
Dopaminergic (SNc) | - | - | - | - | - | -

#### References (parameters)
[1] Lindahl, Mikael, and Jeanette Hellgren Kotaleski. "Untangling basal ganglia network dynamics and function: Role of dopamine depletion and inhibition investigated in a spiking network model." eneuro 3.6 (2016).

[2] Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

[3] Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.

[4] Thibeault, Corey M., and Narayan Srinivasa. "Using a hybrid neuron in physiologically inspired models of the basal ganglia." Frontiers in computational neuroscience 7 (2013): 88.

--- 

McGregor, Matthew M., and Alexandra B. Nelson. "Circuit mechanisms of Parkinson’s disease." Neuron 101.6 (2019): 1042-1056.

Foster, Nicholas N., et al. "The mouse cortico–basal ganglia–thalamic network." Nature 598.7879 (2021): 188-194.

Giordano, Nadia, et al. "Fast-spiking interneurons of the premotor cortex contribute to initiation and execution of spontaneous actions." Journal of Neuroscience 43.23 (2023): 4234-4250.

Chakravarty, Kingshuk, et al. "Transient response of basal ganglia network in healthy and low-dopamine state." eneuro 9.2 (2022).

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.

Fieblinger, Tim. "Striatal control of movement: a role for new neuronal (Sub-) populations?." Frontiers in human neuroscience 15 (2021): 697284.

Dong, Jie, et al. "Connectivity and functionality of the globus pallidus externa under normal conditions and Parkinson's disease." Frontiers in neural circuits 15 (2021): 645287.

The role of neurotransmitter systems in mediating deep brain stimulation effects in Parkinson’s disease

Humphries, Mark D., et al. "Capturing dopaminergic modulation and bimodal membrane behaviour of striatal medium spiny neurons in accurate, reduced models." Frontiers in computational neuroscience 3 (2009): 849.
