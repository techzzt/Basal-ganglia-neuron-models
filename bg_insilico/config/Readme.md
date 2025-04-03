```
# degradation
"target_rates": {
    "FSN": {
        "equation": "((1 - (t / second)) * (646 / 1000)) * Hz"
    },
    "MSND1": {
        "equation": "((1 - (t / second)) * (448 / 1000)) * Hz"
    },
    "MSND2": {
        "equation": "((1 - (t / second)) * (592 / 1000)) * Hz"
    },
    "STN": {
        "equation": "((1 - (t / second)) * (170 / 1000)) * Hz"
    }
}

```

```
"target_rates": {
    "FSN": {
        "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 646*Hz"
    },
    "MSND1": {
        "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 448*Hz"
    },
    "MSND2": {
        "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 592*Hz"
    },
    "STN": {
        "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 170*Hz"
    }
}
```

```
"neuron_type": "poisson",
"target_rates": {
    "FSN": {
        "equation": "((t / second) / 1000 * 646) * Hz"
    },
    "MSND1": {
        "equation": "((t / second) / 1000 * 448) * Hz"
    },
    "MSND2": {
        "equation": "((t / second) / 1000 * 592) * Hz"
    },
    "STN": {
        "equation": "((t / second) / 1000 * 170) * Hz"
    }
}
```     
