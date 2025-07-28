# Basal Ganglia In Silico Simulation

This repository contains a computational model of the basal ganglia network implemented using Brian2 neural simulator.

## 🧠 Overview

The basal ganglia is a group of subcortical nuclei involved in motor control, learning, and decision-making. This simulation models the key components including:

- **MSN D1/D2** (Medium Spiny Neurons)
- **STN** (Subthalamic Nucleus) 
- **GPe** (Globus Pallidus externa)
- **SNr** (Substantia Nigra pars reticulata)
- **FSN** (Fast Spiking Interneurons)

## 🚀 Quick Start

### 1. Environment Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configuration

All simulation parameters are configured in JSON files located in the `config/` directory:

- **Neuron parameters**: Population sizes, model types, parameters
- **Connection parameters**: Synaptic weights, connection probabilities
- **Simulation settings**: Duration, time steps, stimulus configuration
- **Dopamine conditions**: Normal vs dopamine-depleted states

### 3. Running Simulations

Navigate to the `bg_insilico/` directory and run:

```bash
# List available configuration files
python run.py --list-configs

# Run with default configuration (dopamine-depleted)
python run.py

# Run with specific configuration
python run.py --config config/test_normal_noin.json
python run.py -c config/test_dop_noin.json
```

## 📁 Project Structure

```
bg_insilico/
├── config/                 # Configuration files
│   ├── test_dop_noin.json  # Dopamine-depleted condition
│   └── test_normal_noin.json # Normal condition
├── module/
│   ├── models/            # Neuron and synapse models
│   ├── simulation/        # Simulation runner
│   └── utils/            # Utilities and visualization
├── Neuronmodels/         # Neuron model implementations
├── params_ref/           # Reference parameter files
├── run.py               # Main simulation script
└── requirements.txt     # Python dependencies
```

## ⚙️ Configuration Options

### Available Configurations

- `test_dop_noin.json`: Dopamine-depleted condition simulation
- `test_normal_noin.json`: Normal dopamine condition simulation

### Key Parameters

You can modify various parameters in the configuration files:

- **Neuron populations**: Size and types of each neural population
- **Synaptic connections**: Connection probabilities and weights
- **External inputs**: Cortical and external input rates
- **Simulation duration**: Time length and resolution
- **Stimulus parameters**: Timing and intensity of external stimulation

## 📊 Output and Visualization

The simulation generates:

1. **Raster plots**: Spike timing across all neurons
2. **Membrane potential traces**: Voltage dynamics of selected neurons
3. **Firing rate analysis**: Population activity over time
4. **FFT spectra**: Frequency analysis of neural activity
5. **Zoom plots**: Detailed views of specific time windows

## 🔧 Advanced Usage

### Command Line Options

```bash
# Show help
python run.py --help

# List all available configuration files
python run.py --list-configs

# Run with custom configuration
python run.py --config path/to/your/config.json
```

### Customizing Parameters

To modify simulation parameters:

1. Copy an existing configuration file
2. Edit the parameters as needed
3. Run with your custom configuration

Example parameter modifications:
```json
{
  "neurons": [
    {
      "name": "MSND1",
      "N": 37971,
      "params_file": "./params_ref/MSN_D1_1_dop.json"
    }
  ],
  "simulation": {
    "duration": 10000,
    "stimulus": {
      "enabled": true,
      "start_time": 6000,
      "duration": 1000
    }
  }
}
```

## 📚 References

This implementation is based on the following research:

- **Primary Reference**: "Untangling Basal Ganglia Network Dynamics and Function: Role of Dopamine Depletion and Inhibition Investigated in a Spiking Network Model"
- **Brian2 Documentation**: https://brian2.readthedocs.io/
- **Synaptic Currents**: Reference implementation from [Brian2 Discourse](https://brian.discourse.group/t/problem-with-biexponential-synaptic-currents-no-currents-recorded/737)

## 🤝 Contributing

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with different configurations
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Troubleshooting

### Common Issues

1. **Import errors**: Ensure all requirements are installed
2. **Memory issues**: Reduce simulation duration or neuron counts
3. **Slow performance**: Use smaller network sizes for testing

### Getting Help

If you encounter issues:

1. Check the configuration file syntax
2. Verify all required files are present
3. Review the error messages for specific issues
4. Open an issue on GitHub with detailed information

---

**Note**: This simulation is for research purposes. Results should be interpreted in the context of the specific model parameters and assumptions used.
