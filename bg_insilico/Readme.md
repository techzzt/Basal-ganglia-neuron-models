bg_insilico/
├── __init__.py
├── main.py                     # 메인 실행 파일
├── module/
│   ├── __init__.py
│   ├── models/                 # 모델 관련 파일들
│   │   ├── __init__.py
│   │   ├── neuron_models.py   # 뉴런 모델 클래스들
│   │   └── synapse_models.py  # 시냅스 기본 클래스
│   │
│   ├── utils/                 # 유틸리티 함수들
│   │   ├── __init__.py
│   │   ├── plot_utils.py     # plotting 관련 함수들
│   │   └── simulation_utils.py # 시뮬레이션 관련 유틸리티
│   │
│   └── simulation/           # 시뮬레이션 관련 파일들
│       ├── __init__.py
│       └── network.py        # 네트워크 구성 관련
│
├── Neuronmodels/             # 구체적인 뉴런/시냅스 구현
│   ├── __init__.py
│   ├── GPe_STN_inh_ext_dop_nor.py
│   └── GPe_STN_inh_ext_dop_dep.py
│
└── results/                  # 결과 저장 디렉토리
    ├── raster_plots/         # raster plot 저장 위치
    └── other_results/        # 기타 결과 저장 위치
