Date | work |
--- | --- | 
LIF | $$\frac{dv}{dt} = \frac{-g_L \cdot 1 \, \text{pF/ms/mV} \cdot (v - E_L) + I}{C}$$ | 
QIF | $$\frac{dv}{dt} = \frac{k \cdot 1 \, \text{pF/ms/mV} \cdot (v - v_r) \cdot (v - v_t) - u \, \text{pF} + I}{C}$$ <br> <br> $$\frac{du}{dt} = a \cdot \left(b \cdot (v - v_r) - u\right)$$ | $
AdEx | $$C_m \frac{dV}{dt} = -g_L(V - E_L) + g_L \Delta_T e^{\frac{V-V_T}{\Delta_T}} - u + I$$ <br> <br> $$\tau_w \frac{du}{dt} = a(V - E_L) - u$$ | v>t<sup>f</sup>, v = c, u = u + d
Izhikevich |$$\frac{dv}{dt} = \left(0.04 \, \frac{1}{\text{ms} \cdot \text{mV}}\right)v^2 + \left(5 \, \frac{1}{\text{ms}}\right)v + 140 \,\frac{\text{mV}}{\text{ms}} - u + I \quad$$ <br> <br> $$\frac{du}{dt} = a \left(bv - u\right) \quad$$ |

