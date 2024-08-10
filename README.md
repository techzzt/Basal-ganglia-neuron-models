# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)

#### Instantiating a LIF model network:

### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters
--- | --- | ---
LIF | `v` | Membrane potential 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$   $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g_L, Delta_T, E_L, v, vr, vt, tau_w, a, d, C 

- param1.json: Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model (table2, FSN)
- param2.json: Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model (table4, MSN)
- Transient Response of Basal Ganglia Network in Healthy and Low-Dopamine State

**SPN simulation** 
- set1: Lindahl, Mikael, and Jeanette Hellgren Kotaleski. "Untangling basal ganglia network dynamics and function: Role of dopamine depletion and inhibition investigated in a spiking network model." eneuro 3.6 (2016).
- set2: https://github.com/zfountas/basal-ganglia-model/blob/master/README.md
- set3: Gigi, Ilaria, Rosa Senatore, and Angelo Marcelli. "The onset of motor learning impairments in Parkinson’s disease: a computational investigation." Brain Informatics 11.1 (2024): 4.


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



#### Model Information (New)
Neuron | Characteristics | References 
--- | --- | --- 
SPN | activation of dSPN and suppression of iSPN, not co-activation <br/> DA release associated with explicit or implicit reward that active glutamatergic synapse on dSPN and decrease the strength of those on iSPN | [Sheng et al. (2019)](https://pubmed.ncbi.nlm.nih.gov/31072930/) <br/> [Surmeier et al. 2023](https://www.frontiersin.org/journals/synaptic-neuroscience/articles/10.3389/fnsyn.2023.1186484/full)


#### References 
- Kim et al. (2024): Kim, Sang-Yoon, and Woochang Lim. "Quantifying harmony between direct and indirect pathways in the basal ganglia: healthy and Parkinsonian states." Cognitive Neurodynamics (2024): 1-21.
- Sheng, M. J., Lu, D., Shen, Z. M., and Poo, M. M. (2019). Emergence of stable striatal D1R and D2R neuronal ensembles with distinct firing sequence during motor learning. Proc. Natl. Acad. Sci. U. S. A. 116, 11038–11047. doi: 10.1073/pnas.1901712116
- Surmeier, Dalton James, et al. "Rethinking the network determinants of motor disability in Parkinson’s disease." Frontiers in Synaptic Neuroscience 15 (2023): 1186484.
- Kumaravelu, Karthik, David T. Brocker, and Warren M. Grill. "A biophysical model of the cortex-basal ganglia-thalamus network in the 6-OHDA lesioned rat model of Parkinson’s disease." Journal of computational neuroscience 40 (2016): 207-229.
- McCarthy, M. M., et al. "Striatal origin of the pathologic beta oscillations in Parkinson's disease." Proceedings of the national academy of sciences 108.28 (2011): 11620-11625.
- Ortone, Andrea, et al. "Dopamine depletion leads to pathological synchronization of distinct basal ganglia loops in the beta band." PLoS computational biology 19.4 (2023): e1010645.

#### Future work (8/12)
