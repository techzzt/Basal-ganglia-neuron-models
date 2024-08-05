# Basal-ganglia-neuron-models

Hjorth, JJ Johannes, et al. "The microcircuits of striatum in silico." Proceedings of the National Academy of Sciences 117.17 (2020): 9554-9565.


**Network Structure**

![readme](https://github.com/techzzt/Basal-ganglia-neuron-models/assets/49440852/23c7f1c7-f357-4d2a-b546-126fd4996887)

#### Instantiating a LIF model network:

### Equation/Parameters & Variable names:

Eqns | eqs | Hyperparameters
--- | --- | ---
LIF | `v` | Membrane potential 
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$   $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | g_L, Delta_T, E_L, v. vr, vt, tau_w, a, d, C 
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

- param1.json: Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model (table2, FSN)
- param2.json: Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model (table4, MSN)


**SPN simulation** 
- set1: Lindahl, Mikael, and Jeanette Hellgren Kotaleski. "Untangling basal ganglia network dynamics and function: Role of dopamine depletion and inhibition investigated in a spiking network model." eneuro 3.6 (2016).
- set2: https://github.com/zfountas/basal-ganglia-model/blob/master/README.md
- set3: Gigi, Ilaria, Rosa Senatore, and Angelo Marcelli. "The onset of motor learning impairments in Parkinson’s disease: a computational investigation." Brain Informatics 11.1 (2024): 4.


## ARGUMENTS
neuron models should be put under ```./bg_insilico```

- SPN.py define quadratic integrate and fire model 
- simulation.py: generate neuron model, spike mornitoring, plotting 
- synapse.py: generate synapse, connection
- test.ipynb: define params, simulation
  
#### Neuron Models
Based on previous research, construct cortex-striatum 
- SPN: Striatal Projection Neurons
- FS: Fast-spiking Interneurons

#### Neuron Model Features
Neuron | Brain region | Description | References 
--- | --- | --- | ---
STN | Striatum | get input from cortex through the input nuclei | [Kim et al. (2024)](https://link.springer.com/article/10.1007/s11571-024-10119-8)
SNr | Striatum | make inhibitory output projection through the output nucleus <br/> received excitatory input from one CTX neuron, modeled using an alpha synapse | [Kim et al. (2024)](https://link.springer.com/article/10.1007/s11571-024-10119-8) <br/> [Kumaravelu et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26867734/) 
MSN | Striatum | connect primarily to other MSNs through GABAergic synapses and have different rate of connectivity (all, 30% random) | [Mccarthy et al. (2012](https://www.researchgate.net/publication/51242523_Striatal_origin_of_the_pathologic_beta_oscillations_in_Parkinson%27s_disease) 
FSI | Striatum | receive synaptic inputs from RS (regular spiking) neurons | [Kumaravelu et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26867734/) 
GPe | Striatum | receive inhibitory input from all indirect Str MSNs, about 89%-90% of the total connections found in GPe | [Kumaravelu et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26867734/) 
GPi | Striatum | early excitation of GPi was due to activation of STN neurons via the hyperdirect pathway | [Kumaravelu et al. (2016)](https://pubmed.ncbi.nlm.nih.gov/26867734/)


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

#### Future work (8/5)
Laqicque's LIF Neuron Model: https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_2.html#the-passive-membrane

1) 논문 참고해서 새로운 네트워크 추가
2) 새로운 네트워크와 기존 SPN 네트워크 연결
3) 1, 2 이후에 입력 값에 따른 패턴 비교 
