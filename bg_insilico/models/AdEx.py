from brian2 import *

class NeuronModel:
    def __init__(self, N, params):
        self.N = N
        self.params = params
        self.build_model()

    def build_model(self):
        eqs = '''
        dv/dt = (-g_L*(v-E_L) + g_L*Delta_T*exp((v-vt)/Delta_T) - u + I)/C : volt
        du/dt = (a*(v-E_L) - u)/tau_w : volt / second
        I : amp
        g_L      : siemens
        Delta_T  : volt
        vt      : volt
        vr      : volt
        E_L      : volt
        tau_w    : second
        a        : 1/second
        d       : volt/second
        C        : farad
        '''

        self.neurons = NeuronGroup(self.N, model=eqs, threshold='v > vt', reset='v = E_L; u += d', method='euler')
        self.set_parameters()
        
    def set_parameters(self):
        param_values = {}
        for param, info in self.params.items():
            if param == 'N':
                continue  # N은 이미 생성 시 설정되었으므로 건너뜁니다.

            try:
                value = info['value']
                unit = info.get('unit', '')  # unit이 없으면 빈 문자열 사용
                if unit:
                    value = value * eval(unit)  # 단위를 곱합니다.
                param_values[param] = value  # 딕셔너리에 추가
            except Exception as e:
                print(f"Error processing parameter '{param}': {e}")

        # 변수에 직접 값 할당
        for param, value in param_values.items():
            if param in self.neurons.variables:
                self.neurons.variables[param] = value
            else:
                print(f"Warning: '{param}' is not a valid state variable in the NeuronGroup.")


# R. Naud, N. Marcille, C. Clopath, and W. Gerstner, “Firing patterns in the adaptive exponential integrateand-fire model.,” Biol. Cybern., vol. 99, no. 4–5, pp.335–347, Nov. 2008. 