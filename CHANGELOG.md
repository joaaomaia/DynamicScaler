# Changelog

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
