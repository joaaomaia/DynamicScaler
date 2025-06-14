# Changelog

## 0.5.1
- Campo `reason` populado para scalers aceitos com flags de validação.
- Tabela de colunas do report atualizada.

## 0.5.0
- Validação de importância via SHAP/gain substitui ganho de CV.
- Novos parâmetros `importance_metric` e `importance_gain_thr`.
- `cv_gain_thr` depreciado, mapeado para `importance_gain_thr`.
- Documentação e fluxograma atualizados.

## 0.4.0
- Validação secundária via curtose e etapa opcional de cross-validation.
- `MinMaxScaler` apenas se `allow_minmax=True` e sujeito à CV.
- Novos parâmetros `extra_validation`, `allow_minmax`, `kurtosis_thr` e `cv_gain_thr`.
- Serialização salva apenas scalers efetivamente selecionados.
- Documentação atualizada com fluxograma da nova validação.

## 0.3.0
- Refatoração do `DynamicScaler` com validação pós-transform.
- Suporte a `ignore_scalers` e lista de fallback curta.
- Novo parâmetro `validation_fraction` e função de `scoring` customizável.
- Atualização da documentação.

## 0.3.1
- `plot_histograms` agora exibe o scaler a partir de `chosen_scaler`.
- Mensagem "Nenhum" quando a coluna não é escalonada.
- Log adicional durante a plotagem de histogramas.

## 0.3.2
- Teste de normalidade via Shapiro-Wilk para habilitar o `StandardScaler`.
- `MinMaxScaler` adicionado como último candidato no modo `auto`.
- Novos parâmetros `shapiro_n` e `shapiro_p_val` documentados.
