"target_rates": {
    "FSN": {
        "equation": "((t / second) * (646 / 1000) + 3 * randn()) * Hz"
    },
    "MSND1": {
        "equation": "((t / second) * (448 / 1000) + 3 * randn()) * Hz"
    },
    "MSND2": {
        "equation": "((t / second) * (592 / 1000) + 3 * randn()) * Hz"
    },
    "STN": {
        "equation": "((t / second) * (170 / 1000) + 3 * randn()) * Hz"
    }
}
}


            "neuron_type": "poisson",
            "target_rates": {
                "FSN": {
                    "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 646*Hz + 3*Hz * randn()"
                },
                "MSND1": {
                    "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 448*Hz + 3*Hz * randn()"
                },
                "MSND2": {
                    "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 592*Hz + 3*Hz * randn()"
                },
                "STN": {
                    "equation": "0*Hz + (t >= 200*ms) * (t < 400*ms) * 170*Hz + 3*Hz * randn()"
                }
            }
}


            "neuron_type": "poisson",
            "target_rates": {
                "FSN": {
                    "equation": "((t / second) / 1000 * 646 + 3 * randn()) * Hz"
                },
                "MSND1": {
                    "equation": "((t / second) / 1000 * 448 + 3 * randn()) * Hz"
                },
                "MSND2": {
                    "equation": "((t / second) / 1000 * 592 + 3 * randn()) * Hz"
                },
                "STN": {
                    "equation": "((t / second) / 1000 * 170 + 3 * randn()) * Hz"
                }
            }
        }
