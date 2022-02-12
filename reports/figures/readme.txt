santos_v0:
previsao de 40 steps, arquitetura classica, ativacao: tanh, boia de santos, previsao de erro, apenas membros do Hs como features, usando membros do NOAA

santos_v1:
previsao de 80 steps, arquitetura classica, ativacao: tanh, boia de santos, previsao de erro, apenas membros do Hs como features, usando membros do NOAA

santos_v2:
previsao de 40 steps, arquitetura classica, ativacao: relu, boia de santos,previsao de erro, apenas membros do Hs como features, usando membros do NOAA

santos_v3:
previsao de 20 steps, arquitetura classica, ativacao: relu, boia de santos, previsao de erro, apenas membros do Hs como features, usando membros do NOAA

rg_v0:
previsao de 120 steps, arquitetura classica, ativacao: relu, boia de rg, previsao de erro, apenas membros do Hs como features, usando ensembles do ERA5
obs.: com a ativacao 'tanh', nao tem muita diferenca de resultado, mas e bem mais rapido

rg_v1:
previsao de 120 steps, arquitetura classica, ativacao: tanh, boia de rg, previsao de erro, apenas membros do Hs como features, usando ensembles do ERA5, previsao do futuro, e nao lead a lead.

rg_v2:
previsao de 40 steps, arquitetura classica, ativacao: tanh, boia de rg, previsao de erro, apenas membros do Hs como features, usando ensembles do ERA5, previsao do futuro, e nao lead a lead.

rg_v3:
previsao de 480 steps, arquitetura classica, ativacao: tanh, boia de rg, previsao de erro, apenas membros do Hs como features, usando ensembles do ERA5, previsao do futuro, e nao lead a lead.

santos_v4:
previsao de 40 steps, arquitetura classica, ativacao: tanh, boia de santos, previsao de erro, apenas membros do Hs como features, usando ensembles do NOAA, previsao do futuro, e nao lead a lead.

santos_v5:
previsao de 120 steps, arquitetura classica, ativacao: tanh, boia de santos, previsao de erro, apenas membros do Hs como features, usando ensembles do NOAA, previsao do futuro, e nao lead a lead - deterministic

santos_v6:
previsao de 40 steps, arquitetura classica, ativacao: tanh, boia de santos, previsao de erro, apenas membros do Hs como features, usando ensembles do NOAA, previsao do futuro, e nao lead a lead - deterministic
