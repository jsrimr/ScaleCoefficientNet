# Scale coefficient Exploration

Phase 1. Sample networks from largeNet
    ```
    git submodule init
    cd tests
    PYTHONPATH=../ python tests/run_subnets.py
    ```
    => ../results 에 네트워크별 acc, latency 가 pickle 로 저장됨
    
    Requirements
    - DataProvider (test : tests/test_input_pipeline.py)
    - Train loop (test : tests/test_train_loop.py)
    (위에서 Sample 뽑는데는 이번에는 사용되진 않음.) 




