# Basal Ganglia In Silico Simulation

A computational model of the basal ganglia network using Brian2 neural simulator.

## Start

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

## Key Files

- `config/`: Simulation configuration files
- `module/`: Neuron models, simulation, and visualization code
- `run.py`: Main execution script

## Configuration

You can modify the following in JSON files:
- Neuron population sizes and parameters
- Synaptic connection probabilities and weights
- External input rates
- Simulation duration and resolution

## Output

The simulation generates:
- Spike patterns (raster plot)  
<p align="center">
  <img width="500" alt="Spike Raster" src="https://github.com/user-attachments/assets/af89b479-dfff-45d9-bc0b-895bd2978160" />
</p>

- Membrane potential changes  
<p align="center">
  <img width="500" alt="Membrane Potential" src="https://github.com/user-attachments/assets/c6b32c48-841e-4f43-8e34-3172bd0e91fc" />
</p>

- Firing rate analysis  
<p align="center">
  <img width="500" alt="Firing Rate" src="https://github.com/user-attachments/assets/66d53a2a-d897-43e5-94a5-f88c9f16394b" />
</p>

- FFT spectrum  
<p align="center">
  <img width="500" alt="FFT Spectrum" src="https://github.com/user-attachments/assets/d4b88ea1-2497-40a1-ae43-9fca599b3fa5" />
</p>


## 📚 References

- "Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model"
- [Brian2 Documentation](https://brian2.readthedocs.io/)

