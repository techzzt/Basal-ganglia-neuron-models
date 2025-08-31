# Basal Ganglia In Silico Simulation

A computational model of the basal ganglia network using Brian2 neural simulator.

## 🚀 Quick Start

### 1. Setup Environment
```bash
pip install -r requirements.txt
```

### 2. Run Simulation
```bash
python run.py --list-configs
python run.py
python run.py --config config/test_normal_noin.json
```

## 📁 Key Files

- `config/`: Simulation configuration files
- `module/`: Neuron models, simulation, and visualization code
- `run.py`: Main execution script

## ⚙️ Configuration

You can modify the following in JSON files:
- Neuron population sizes and parameters
- Synaptic connection probabilities and weights
- External input rates
- Simulation duration and resolution

## 📊 Output

The simulation generates:
- Spike patterns (raster plot)
<img width="590" height="454" alt="image" src="https://github.com/user-attachments/assets/af89b479-dfff-45d9-bc0b-895bd2978160" />

- Membrane potential changes
  <img width="320" height="125" alt="image" src="https://github.com/user-attachments/assets/c6b32c48-841e-4f43-8e34-3172bd0e91fc" />

- Firing rate analysis
  <img width="372" height="233" alt="image" src="https://github.com/user-attachments/assets/66d53a2a-d897-43e5-94a5-f88c9f16394b" />

- FFT spectrum
<img width="313" height="227" alt="image" src="https://github.com/user-attachments/assets/d4b88ea1-2497-40a1-ae43-9fca599b3fa5" />


## 📚 References

- "Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model"
- [Brian2 Documentation](https://brian2.readthedocs.io/)

