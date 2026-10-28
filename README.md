# VP2 · Aprendizado de Máquina (PPGEP 2026.2)

Repositório com todos os entregáveis da VP2: classificação de defeitos com CNN e modelos clássicos (Casting Product e NEU Surface Defect) e previsão de RUL com RNN (C-MAPSS FD001).

## Principais resultados

| Tarefa | Melhor modelo | Métrica no teste |
|---|---|---|
| Casting (ok × defeito) | CNN | F1 0,996 |
| NEU (6 classes) | CNN | F1 0,993 (SVM 0,921) |
| C-MAPSS FD001 (RUL) | GRU 2×96, janela 50 | RMSE 14,54 · MAE 10,83 · NASA 379 |

## Estrutura

- `notebooks/VP2_CNN_Casting_NEU.ipynb`: Parte 1 (Casting + NEU; MLP, SVM, AD, RF, NB e CNN; GridSearch × Optuna), já executado.
- `notebooks/VP2_RNN_CMAPSS.ipynb`: Parte 2 (C-MAPSS FD001, RUL com SimpleRNN/LSTM/GRU, Optuna TPE + Hyperband), já executado.
- `notebooks/*.py`: os mesmos notebooks em formato .py (jupytext).
- `modelos/`: `cnn_casting.keras`, `cnn_neu.keras`, `rnn_rul_fd001_best.keras` (melhor modelo RUL) e `rnn_baseline.keras`. Para carregar: `keras.models.load_model(caminho)`.
- `resultados/`: tabelas em CSV (métricas, comparação GridSearch × Optuna, trials do Optuna).
- `relatorio/`: relatório completo (`.pdf`, `.md`, `.html` com as figuras embutidas, `.docx`) e a pasta `figuras/`.

## Como rodar de novo (por exemplo, no Google Colab)

1. Baixe os datasets do Kaggle e organize assim dentro de uma pasta `data/`:
   - `data/casting_data/{train,test}/{def_front,ok_front}/`
   - `data/NEU/{train,valid,test}/{Crazing,Inclusion,Patches,Pitted,Rolled,Scratches}/` (versão fantacher/neu-metal-surface-defects-data)
   - `data/CMAPSS/{train_FD001.txt,test_FD001.txt,RUL_FD001.txt}`
2. `pip install tensorflow optuna scikit-learn scikit-image seaborn`
3. Se a pasta estiver em outro lugar, defina `VP2_DATA=/caminho/para/data`.

Tempo em CPU (4 núcleos): cerca de 4 h para o notebook CNN e 1,5 h para o RNN. Com GPU, o tempo cai bastante.
