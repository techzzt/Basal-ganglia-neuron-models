# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)


### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters
--- | --- | ---
LIF | $$\frac{dv}{dt} = \frac{-g_L \cdot 1 \, \text{pF/ms/mV} \cdot (v - E_L) + I}{C}$$ | g_L, E_L, d, vr, vt, C
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$
$$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | g_L, Delta_T, E_L, v, vr, vt, tau_w, a, d, C 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$   $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g_L, Delta_T, E_L, v, vr, vt, tau_w, a, d, C 

- Transient Response of Basal Ganglia Network in Healthy and Low-Dopamine State


## ARGUMENTS
neuron models should be put under ```./bg_insilico```

- models/model{}.py: define quadratic integrate and fire model (AdEx, LIF, QIF, HH, Izh)
- params: parameter list with json format
- simulation_runner.py: generate neuron model, spike mornitoring
- result.py: plotting 
- test.ipynb: define params, simulation


#### Neuron Model Features
Neuron type | Brain region | Description | References | Alternative terms | Neuron transmitter
--- | --- | --- | --- | --- | ---
MSN D1 | Striatum |  | | | 
MSN D2 | Striatum |  | | | 
FSN | Striatum |  | | | 
ChiN | Striatum |  | | | 
prototypic | GPe |  | | | 
Arkypallidal | GPe |  | | | 
PV+ | STN |  | | | 
PV- | STN |  | | | 
PV+ | GPi |  | | | 
SST+ | GPi |  | | | 
PV+ | SNr |  | | | 
SST+ | SNr |  | | | 
GABAergic | SNc |  | | | 
Dopaminergic | SNc |  | | | 


#### References 
- Kim et al. (2024): Kim, Sang-Yoon, and Woochang Lim. "Quantifying harmony between direct and indirect pathways in the basal ganglia: healthy and Parkinsonian states." Cognitive Neurodynamics (2024): 1-21.
- Sheng, M. J., Lu, D., Shen, Z. M., and Poo, M. M. (2019). Emergence of stable striatal D1R and D2R neuronal ensembles with distinct firing sequence during motor learning. Proc. Natl. Acad. Sci. U. S. A. 116, 11038–11047. doi: 10.1073/pnas.1901712116
- Surmeier, Dalton James, et al. "Rethinking the network determinants of motor disability in Parkinson’s disease." Frontiers in Synaptic Neuroscience 15 (2023): 1186484.
- Kumaravelu, Karthik, David T. Brocker, and Warren M. Grill. "A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease." Journal of computational neuroscience 40 (2016): 207-229.
- McCarthy, M. M., et al. "Striatal origin of the pathologic beta oscillations in Parkinson's disease." Proceedings of the national academy of sciences 108.28 (2011): 11620-11625.
- Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.
