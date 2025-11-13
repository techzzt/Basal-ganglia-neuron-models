# CBTC circuit Simulation

A computational model of the basal ganglia network using Brian2 library

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
- Membrane potential changes  

## References

- "Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model"
- [Brian2 Documentation](https://brian2.readthedocs.io/)

