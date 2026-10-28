import os
import shutil
import subprocess
import webbrowser
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

from plotly.colors import sample_colorscale
from plotly.offline import get_plotlyjs
from plotly.subplots import make_subplots

######### ETAPA 1 — CARREGAMENTO #########

base_url = "https://siros.anac.gov.br/siros/registros/diversos/vra/2025/"

arquivos = [f"VRA_2025_{mes:02d}.csv" for mes in range(1, 13)]
dfs = []
estruturas_arquivos = {}

for arquivo in arquivos:
    print(f"Carregando {arquivo}...")

    df_mes = pd.read_csv(
        base_url + arquivo,
        sep=";",
        encoding="utf-8",
        low_memory=False
    )

    estruturas_arquivos[arquivo] = {
        "shape": df_mes.shape,
        "colunas": df_mes.columns.tolist()
    }

    dfs.append(df_mes)

df = pd.concat(dfs, ignore_index=True, sort=False)

print("\n" + "=" * 70)
print("ETAPA 1 — CARREGAMENTO CONCLUÍDO")
print("=" * 70)
print(f"Arquivos carregados: {len(dfs)} de {len(arquivos)}")
print(f"Dimensões: {df.shape[0]:,} linhas x {df.shape[1]} colunas")

######### ETAPA 2 — INSPEÇÃO #########
# Objetivo: conhecer estrutura, conteúdo e distribuição

######### 2.1 — Dimensões e visualização inicial #########

print("\n--- DIMENSÕES DA BASE ---")
print(df.shape)

print("\n--- PRIMEIRAS LINHAS ---")
print(df.head())

print("\n--- ÚLTIMAS LINHAS ---")
print(df.tail())

######### 2.2 — Tipos de dados e informações gerais #########

print("\n--- TIPOS DE DADOS ---")
print(df.dtypes)

print("\n--- INFORMAÇÕES DO DATAFRAME ---")
df.info()

######### 2.3 — Inspeção inicial de valores ausentes #########

print("\n--- QUANTIDADE DE VALORES AUSENTES POR COLUNA ---")
nulos = df.isnull().sum()
print(nulos)

print("\n--- PERCENTUAL DE VALORES AUSENTES POR COLUNA ---")
percentual_nulos = df.isnull().mean().mul(100)
print(percentual_nulos.sort_values(ascending=False).round(3))

######### 2.4 — Inspeção inicial de duplicidades #########

print("\n--- CÓPIAS EXCEDENTES DUPLICADAS ---")
print(f"Quantidade: {df.duplicated().sum():,}")

duplicados_df = df.loc[df.duplicated(keep=False)].copy()

print("\n--- REGISTROS ENVOLVIDOS EM DUPLICIDADES ---")

if duplicados_df.empty:
    print("Nenhum registro duplicado encontrado.")
else:
    colunas_duplicados = [
        "Sigla ICAO Empresa Aérea",
        "Empresa Aérea",
        "Número Voo",
        "Referência",
        "Modelo Equipamento",
        "Número de Assentos",
        "Sigla ICAO Aeroporto Origem",
        "Descrição Aeroporto Origem",
        "Partida Prevista",
        "Partida Real",
        "Sigla ICAO Aeroporto Destino",
        "Descrição Aeroporto Destino",
        "Chegada Prevista",
        "Chegada Real",
        "Situação Voo",
        "Situação Partida",
        "Situação Chegada"
    ]

    colunas_disponiveis = [
        coluna for coluna in colunas_duplicados
        if coluna in duplicados_df.columns
    ]

    print(duplicados_df[colunas_disponiveis].to_string(index=True))

    # O agrupamento inclui Referência para diferenciar datas distintas.
    print("\n--- GRUPOS POR EMPRESA, VOO, DATA E ROTA ---")

    colunas_grupo = [
        "Sigla ICAO Empresa Aérea",
        "Número Voo",
        "Referência",
        "Sigla ICAO Aeroporto Origem",
        "Sigla ICAO Aeroporto Destino"
    ]

    print(
        duplicados_df
        .groupby(colunas_grupo, dropna=False)
        .size()
        .sort_values(ascending=False)
    )

    print(f"\nLinhas envolvidas em duplicidades: {len(duplicados_df):,}")

    print("\n--- DUPLICIDADES POR SITUAÇÃO DO VOO ---")
    print(duplicados_df["Situação Voo"].value_counts(dropna=False))

    print("\n--- DUPLICIDADES POR EMPRESA AÉREA ---")
    print(duplicados_df["Empresa Aérea"].value_counts(dropna=False))

######### 2.5 — Inspeção da estrutura dos arquivos mensais #########

colunas_referencia = None
arquivos_com_estrutura_diferente = []

for arquivo, info in estruturas_arquivos.items():
    print(f"\nVerificando {arquivo}...")
    print(f"Dimensões: {info['shape']}")
    print(f"Colunas: {info['colunas']}")

    if colunas_referencia is None:
        colunas_referencia = info["colunas"]
    elif info["colunas"] != colunas_referencia:
        arquivos_com_estrutura_diferente.append(arquivo)

        print("ATENÇÃO: estrutura diferente do primeiro arquivo.")
        print(
            "Colunas ausentes:",
            set(colunas_referencia) - set(info["colunas"])
        )
        print(
            "Colunas adicionais:",
            set(info["colunas"]) - set(colunas_referencia)
        )
        print(
            "Mesmas colunas, possivelmente em ordem diferente:",
            set(info["colunas"]) == set(colunas_referencia)
        )

print("\nArquivos com estrutura diferente:", arquivos_com_estrutura_diferente)
print("Total consolidado:", df.shape)

######### 2.6 — Cobertura temporal dos registros #########

print("\n--- COBERTURA TEMPORAL ---")

referencia = pd.to_datetime(
    df["Referência"],
    format="%d/%m/%Y %H:%M:%S",
    errors="coerce"
)

referencia_original = df["Referência"].astype("string")
referencia_preenchida = (
    referencia_original.notna()
    & referencia_original.str.strip().ne("")
)

referencia_invalida = referencia_preenchida & referencia.isna()

print("Referências originalmente ausentes:", df["Referência"].isna().sum())
print("Referências não vazias que falharam na conversão:", referencia_invalida.sum())
print("Data mínima:", referencia.min())
print("Data máxima:", referencia.max())

print("\n--- QUANTIDADE DE REGISTROS POR MÊS ---")
print(
    referencia.dt.to_period("M")
    .value_counts()
    .sort_index()
)

print("\n--- SITUAÇÃO DOS VOOS POR MÊS ---")
print(
    pd.crosstab(
        referencia.dt.month,
        df["Situação Voo"],
        dropna=False
    ).rename_axis("Mês")
)
######### 2.7 — Distribuição do Código Tipo Linha #########

print("\n--- CÓDIGO TIPO LINHA ---")
print(df["Código Tipo Linha"].value_counts(dropna=False))

print("\n--- PERCENTUAL DO CÓDIGO TIPO LINHA ---")
print(
    df["Código Tipo Linha"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(3)
)

######### 2.8 — Inspeção de textos vazios e espaços extras #########

colunas_texto = df.select_dtypes(
    include=["object", "string"]
).columns

print("\n--- TEXTOS VAZIOS OU COM APENAS ESPAÇOS ---")

for coluna in colunas_texto:
    valores = df[coluna].astype("string")

    vazios = valores.str.strip().eq("").fillna(False).sum()
    espacos = (
        valores.notna()
        & valores.ne(valores.str.strip())
    ).sum()

    if vazios > 0 or espacos > 0:
        print(
            f"{coluna}: "
            f"vazios={vazios:,} | "
            f"valores com espaços extras={espacos:,}"
        )

print("A inspeção não substitui nem remove esses valores.")

######### 2.8.1 Verificação de códigos ICAO vazios #########

for coluna in [
    "Sigla ICAO Aeroporto Origem",
    "Sigla ICAO Aeroporto Destino"
]:
    valores = df[coluna].astype("string").str.strip()

    print(f"\n{coluna}")
    print("Valores NaN:", int(df[coluna].isna().sum()))
    print("Strings vazias:", int(valores.eq("").fillna(False).sum()))


    # Verificação de possíveis códigos de origem fora do padrão

origem = df["Sigla ICAO Aeroporto Origem"].astype("string").str.strip()

print("Valores únicos de origem com representação:")
print(
    origem.value_counts(dropna=False)
    .head(10)
    .to_string()
)

print("\nRegistros com destino SBGL:")
print(
    df.loc[
        df["Sigla ICAO Aeroporto Destino"].astype("string").str.strip().eq("SBGL"),
        [
            "Sigla ICAO Aeroporto Origem",
            "Descrição Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Descrição Aeroporto Destino"
        ]
    ].head(10).to_string(index=False)
)

######### 2.9 — Inspeção dos formatos de data e hora #########

colunas_horarios = [
    "Partida Prevista",
    "Partida Real",
    "Chegada Prevista",
    "Chegada Real"
]

for coluna in colunas_horarios:
    print(f"\n--- {coluna} ---")

    original = df[coluna].astype("string")
    preenchidos = original.notna() & original.str.strip().ne("")

    valores = original.loc[preenchidos].str.strip()

    print("Exemplos:")
    print(valores.head(10).tolist())

    print("Comprimentos dos valores mais frequentes:")
    print(valores.str.len().value_counts().head(10))

    convertidos = pd.to_datetime(
        df[coluna],
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    nao_convertidos = preenchidos & convertidos.isna()

    print(
        "Valores não vazios que falharam na conversão:",
        int(nao_convertidos.sum())
    )

######### 2.10 — Situação geral dos voos #########

print("\n--- SITUAÇÃO DOS VOOS ---")
situacao_voo = df["Situação Voo"].value_counts(dropna=False)
print(situacao_voo)

print("\n--- PERCENTUAL POR SITUAÇÃO DO VOO ---")
percentual_situacao = (
    df["Situação Voo"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
)
print(percentual_situacao.round(3))

print("\n--- INDICADORES GERAIS ---")
total_registros = len(df)
print(f"Total de registros: {total_registros:,}")

for situacao, quantidade in situacao_voo.items():
    percentual = quantidade / total_registros * 100
    print(f"{situacao}: {quantidade:,} ({percentual:.2f}%)")


######### 2.11 — Situação das partidas #########

print("\n--- SITUAÇÃO DA PARTIDA ---")
situacao_partida = df["Situação Partida"].value_counts(dropna=False)
print(situacao_partida)

print("\n--- PERCENTUAL DA SITUAÇÃO DA PARTIDA ---")
print(
    df["Situação Partida"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(3)
)

######### 2.12 — Situação das chegadas #########

print("\n--- SITUAÇÃO DA CHEGADA ---")
situacao_chegada = df["Situação Chegada"].value_counts(dropna=False)
print(situacao_chegada)

print("\n--- PERCENTUAL DA SITUAÇÃO DA CHEGADA ---")
print(
    df["Situação Chegada"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .round(3)
)

######### 2.13 — Cruzamentos entre situação do voo e classificação operacional #########

print("\n--- SITUAÇÃO DO VOO X SITUAÇÃO DA PARTIDA ---")
print(
    pd.crosstab(
        df["Situação Voo"],
        df["Situação Partida"],
        margins=True,
        dropna=False
    )
)

print("\n--- SITUAÇÃO DO VOO X SITUAÇÃO DA CHEGADA ---")
print(
    pd.crosstab(
        df["Situação Voo"],
        df["Situação Chegada"],
        margins=True,
        dropna=False
    )
)

######### 2.14 — Inspeção de horários ausentes por situação do voo #########

for coluna in colunas_horarios:
    print(f"\n--- {coluna.upper()} AUSENTE X SITUAÇÃO DO VOO ---")
    print(
        pd.crosstab(
            df["Situação Voo"],
            df[coluna].isna(),
            margins=True,
            dropna=False
        )
    )

######### 2.15 — Verificação de consistência dos horários reais #########

partida_real_dt = pd.to_datetime(
    df["Partida Real"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

chegada_real_dt = pd.to_datetime(
    df["Chegada Real"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

realizados_sem_partida_real = (
    df["Situação Voo"].eq("REALIZADO")
    & partida_real_dt.isna()
)

cancelados_com_partida_real = (
    df["Situação Voo"].eq("CANCELADO")
    & partida_real_dt.notna()
)

realizados_sem_chegada_real = (
    df["Situação Voo"].eq("REALIZADO")
    & chegada_real_dt.isna()
)

cancelados_com_chegada_real = (
    df["Situação Voo"].eq("CANCELADO")
    & chegada_real_dt.notna()
)

print("\n--- VERIFICAÇÕES DE HORÁRIOS REAIS ---")
print(f"Voos realizados sem partida real: {realizados_sem_partida_real.sum():,}")
print(f"Voos cancelados com partida real: {cancelados_com_partida_real.sum():,}")
print(f"Voos realizados sem chegada real: {realizados_sem_chegada_real.sum():,}")
print(f"Voos cancelados com chegada real: {cancelados_com_chegada_real.sum():,}")

######### 2.16 — Consistência cronológica dos horários #########

partida_prevista_dt = pd.to_datetime(
    df["Partida Prevista"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

chegada_prevista_dt = pd.to_datetime(
    df["Chegada Prevista"],
    format="%d/%m/%Y %H:%M",
    errors="coerce"
)

chegada_antes_partida = (
    partida_real_dt.notna()
    & chegada_real_dt.notna()
    & (chegada_real_dt < partida_real_dt)
)

chegada_prevista_antes_partida = (
    partida_prevista_dt.notna()
    & chegada_prevista_dt.notna()
    & (chegada_prevista_dt < partida_prevista_dt)
)

print("\n--- CONSISTÊNCIA CRONOLÓGICA DOS HORÁRIOS ---")
print(
    "Registros com chegada real anterior à partida real:",
    int(chegada_antes_partida.sum())
)
print(
    "Registros com chegada prevista anterior à partida prevista:",
    int(chegada_prevista_antes_partida.sum())
)

print("\n--- EXEMPLOS COM CHEGADA REAL ANTERIOR À PARTIDA ---")
colunas_exemplo_cronologia = [
    "Empresa Aérea",
    "Número Voo",
    "Referência",
    "Partida Real",
    "Chegada Real",
    "Sigla ICAO Aeroporto Origem",
    "Sigla ICAO Aeroporto Destino"
]

print(df.loc[chegada_antes_partida, colunas_exemplo_cronologia].head(20))

######### 2.17 — Voos realizados sem horário previsto #########

realizados_sem_prevista = df.loc[
    df["Situação Voo"].eq("REALIZADO")
    & (
        partida_prevista_dt.isna()
        | chegada_prevista_dt.isna()
    )
].copy()

print("\n--- VOOS REALIZADOS SEM PARTIDA OU CHEGADA PREVISTA ---")
print(f"Quantidade: {len(realizados_sem_prevista):,}")
print(
    f"Percentual da base: "
    f"{len(realizados_sem_prevista) / len(df) * 100:.3f}%"
)

colunas_horarios_ausentes = [
    "Sigla ICAO Empresa Aérea",
    "Empresa Aérea",
    "Número Voo",
    "Sigla ICAO Aeroporto Origem",
    "Descrição Aeroporto Origem",
    "Partida Prevista",
    "Partida Real",
    "Sigla ICAO Aeroporto Destino",
    "Descrição Aeroporto Destino",
    "Chegada Prevista",
    "Chegada Real",
    "Situação Voo"
]

print(realizados_sem_prevista[colunas_horarios_ausentes].head(10))

######### 2.18 — Horários previstos ausentes por empresa e aeroporto #########

sem_prevista = df.loc[
    df["Situação Voo"].eq("REALIZADO")
    & (
        partida_prevista_dt.isna()
        | chegada_prevista_dt.isna()
    )
].copy()

print("\n--- AUSÊNCIAS POR EMPRESA AÉREA ---")
print(sem_prevista["Empresa Aérea"].value_counts(dropna=False).head(20))

print("\n--- AUSÊNCIAS POR AEROPORTO DE ORIGEM ---")
print(
    sem_prevista["Sigla ICAO Aeroporto Origem"]
    .value_counts(dropna=False)
    .head(20)
)

print("\n--- AUSÊNCIAS POR MODELO DE EQUIPAMENTO ---")
print(sem_prevista["Modelo Equipamento"].value_counts(dropna=False).head(20))

print("\n--- AUSÊNCIAS POR TIPO DE LINHA ---")
print(sem_prevista["Código Tipo Linha"].value_counts(dropna=False))

print("\n--- PARTIDA E CHEGADA PREVISTAS AUSENTES ---")
print(
    pd.crosstab(
        partida_prevista_dt.isna(),
        chegada_prevista_dt.isna(),
        margins=True,
        dropna=False
    )
)

######### 2.19 — Aeroportos de origem #########

print("\n--- QUANTIDADE DE AEROPORTOS DE ORIGEM ---")
print(df["Sigla ICAO Aeroporto Origem"].nunique(dropna=True))

print("\n--- VALORES AUSENTES NO AEROPORTO DE ORIGEM ---")
print(df["Sigla ICAO Aeroporto Origem"].isna().sum())

print("\n--- 20 AEROPORTOS DE ORIGEM COM MAIS REGISTROS ---")
print(df["Sigla ICAO Aeroporto Origem"].value_counts(dropna=False).head(20))

print("\n--- PERCENTUAL DOS 20 PRINCIPAIS AEROPORTOS DE ORIGEM ---")
print(
    df["Sigla ICAO Aeroporto Origem"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .head(20)
    .round(3)
)

######### 2.20 — Aeroportos de destino #########

print("\n--- QUANTIDADE DE AEROPORTOS DE DESTINO ---")
print(df["Sigla ICAO Aeroporto Destino"].nunique(dropna=True))

print("\n--- VALORES AUSENTES NO AEROPORTO DE DESTINO ---")
print(df["Sigla ICAO Aeroporto Destino"].isna().sum())

print("\n--- 20 AEROPORTOS DE DESTINO COM MAIS REGISTROS ---")
print(df["Sigla ICAO Aeroporto Destino"].value_counts(dropna=False).head(20))

print("\n--- PERCENTUAL DOS 20 PRINCIPAIS AEROPORTOS DE DESTINO ---")
print(
    df["Sigla ICAO Aeroporto Destino"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .head(20)
    .round(3)
)

######### 2.21 — Consistência dos códigos e nomes dos aeroportos #########

for codigo, descricao, tipo in [
    (
        "Sigla ICAO Aeroporto Origem",
        "Descrição Aeroporto Origem",
        "ORIGEM"
    ),
    (
        "Sigla ICAO Aeroporto Destino",
        "Descrição Aeroporto Destino",
        "DESTINO"
    )
]:
    print(f"\n--- CONSISTÊNCIA DOS AEROPORTOS: {tipo} ---")

    quantidade_descricoes = (
        df.groupby(codigo, dropna=False)[descricao]
        .nunique(dropna=False)
    )

    inconsistentes = quantidade_descricoes[quantidade_descricoes > 1]

    print(
        "Códigos associados a mais de uma descrição:",
        len(inconsistentes)
    )

    if not inconsistentes.empty:
        print(
            df.loc[
                df[codigo].isin(inconsistentes.index),
                [codigo, descricao]
            ]
            .drop_duplicates()
            .sort_values(codigo, na_position="last")
            .head(50)
        )


######### 2.22 — Principais rotas e quantidade de combinações #########

colunas_rota = [
    "Sigla ICAO Aeroporto Origem",
    "Sigla ICAO Aeroporto Destino"
]

print("\n--- 20 PRINCIPAIS ROTAS (ORIGEM X DESTINO) ---")

rotas = (
    df.groupby(colunas_rota, dropna=False)
    .size()
    .sort_values(ascending=False)
)

print(rotas.head(20))

quantidade_rotas = df[colunas_rota].drop_duplicates().shape[0]

print("\n--- QUANTIDADE DE COMBINAÇÕES ORIGEM-DESTINO ---")
print(quantidade_rotas)

# Verificação adicional: possíveis códigos de origem ou destino vazios
print("\n--- VERIFICAÇÃO DE ROTAS COM CÓDIGOS VAZIOS ---")

for coluna in colunas_rota:
    valores = df[coluna].astype("string").str.strip()

    print(f"\n{coluna}")
    print("Valores NaN:", int(df[coluna].isna().sum()))
    print("Strings vazias:", int(valores.eq("").fillna(False).sum()))

# Conferir as rotas mais frequentes com destino SBGL
print("\n--- 10 ROTAS MAIS FREQUENTES COM DESTINO SBGL ---")

print(
    rotas.loc[
        rotas.index.get_level_values("Sigla ICAO Aeroporto Destino") == "SBGL"
    ].head(10)
)

######### 2.22.1 — Investigação da rota aparentemente sem origem #########

origem = df["Sigla ICAO Aeroporto Origem"].astype("string").str.strip()
destino = df["Sigla ICAO Aeroporto Destino"].astype("string").str.strip()

print("\n--- REGISTROS COM ORIGEM SBGR E DESTINO SBGL ---")
print("Quantidade:", int((origem.eq("SBGR") & destino.eq("SBGL")).sum()))

print("\n--- REGISTROS COM ORIGEM DIFERENTE DE SBGR E DESTINO SBGL ---")
mascara = destino.eq("SBGL") & ~origem.eq("SBGR")

print("Quantidade:", int(mascara.sum()))

print(
    df.loc[
        mascara,
        [
            "Sigla ICAO Aeroporto Origem",
            "Descrição Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Descrição Aeroporto Destino"
        ]
    ].head(10).to_string(index=False)
)

print("\n--- REPRESENTAÇÃO DO ÍNDICE DA ROTA COM 5282 REGISTROS ---")
for (o, d), quantidade in rotas.items():
    if quantidade == 5282 and d == "SBGL":
        print(f"Origem={o!r} | Destino={d!r} | Quantidade={quantidade}")

######### 2.23 — Registros com origem igual ao destino #########

origem_igual_destino = df.loc[
    df["Sigla ICAO Aeroporto Origem"].eq(
        df["Sigla ICAO Aeroporto Destino"]
    )
    & df["Sigla ICAO Aeroporto Origem"].notna()
    & df["Sigla ICAO Aeroporto Destino"].notna()
].copy()

print("\n--- REGISTROS COM ORIGEM IGUAL AO DESTINO ---")
print(f"Quantidade: {len(origem_igual_destino):,}")
print(f"Percentual da base: {len(origem_igual_destino) / len(df) * 100:.3f}%")

print(
    origem_igual_destino[
        [
            "Sigla ICAO Empresa Aérea",
            "Empresa Aérea",
            "Número Voo",
            "Sigla ICAO Aeroporto Origem",
            "Descrição Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Descrição Aeroporto Destino",
            "Situação Voo"
        ]
    ].head(10)
)

######### 2.24 — Investigação de voos com origem igual ao destino #########

print("\n--- SITUAÇÃO DOS REGISTROS ---")
print(origem_igual_destino["Situação Voo"].value_counts(dropna=False))

print("\n--- EMPRESAS AÉREAS MAIS FREQUENTES ---")
print(origem_igual_destino["Empresa Aérea"].value_counts(dropna=False).head(20))

print("\n--- MODELOS DE EQUIPAMENTO MAIS FREQUENTES ---")
print(origem_igual_destino["Modelo Equipamento"].value_counts(dropna=False).head(20))

print("\n--- EXEMPLOS DE REGISTROS ---")
print(
    origem_igual_destino[
        [
            "Empresa Aérea",
            "Número Voo",
            "Modelo Equipamento",
            "Sigla ICAO Aeroporto Origem",
            "Partida Prevista",
            "Partida Real",
            "Situação Voo"
        ]
    ].head(10)
)

######### 2.25 — Situação dos voos por aeroporto #########

print("\n--- SITUAÇÃO DO VOO POR AEROPORTO DE ORIGEM ---")

situacao_por_origem = pd.crosstab(
    df["Sigla ICAO Aeroporto Origem"],
    df["Situação Voo"],
    dropna=False
)

if "REALIZADO" in situacao_por_origem.columns:
    print(
        situacao_por_origem
        .sort_values("REALIZADO", ascending=False)
        .head(20)
    )
else:
    print(situacao_por_origem.head(20))

print("\n--- SITUAÇÃO DO VOO POR AEROPORTO DE DESTINO ---")

situacao_por_destino = pd.crosstab(
    df["Sigla ICAO Aeroporto Destino"],
    df["Situação Voo"],
    dropna=False
)

if "REALIZADO" in situacao_por_destino.columns:
    print(
        situacao_por_destino
        .sort_values("REALIZADO", ascending=False)
        .head(20)
    )
else:
    print(situacao_por_destino.head(20))

######### 2.26 — Empresas aéreas: quantidade e volume #########

print("\n--- QUANTIDADE DE EMPRESAS AÉREAS ---")
print(df["Empresa Aérea"].nunique(dropna=True))

print("\n--- 20 EMPRESAS COM MAIS REGISTROS ---")
print(df["Empresa Aérea"].value_counts(dropna=False).head(20))

print("\n--- PERCENTUAL DAS 20 EMPRESAS COM MAIS REGISTROS ---")
print(
    df["Empresa Aérea"]
    .value_counts(normalize=True, dropna=False)
    .mul(100)
    .head(20)
    .round(3)
)

######### 2.27 — Distribuição do número de assentos #########

assentos = pd.to_numeric(df["Número de Assentos"], errors="coerce")

print("\n--- RESUMO DO NÚMERO DE ASSENTOS ---")
print(assentos.describe())

print("\n--- QUANTIDADE DE VALORES AUSENTES OU NÃO NUMÉRICOS ---")
print(assentos.isna().sum())

print("\n--- QUANTIDADE DE REGISTROS COM ZERO ASSENTOS ---")
print(assentos.eq(0).sum())

print("\n--- QUANTIDADE DE REGISTROS COM ASSENTOS NEGATIVOS ---")
print(assentos.lt(0).sum())

print("\n--- MODELOS COM ZERO ASSENTOS ---")
print(
    df.loc[assentos.eq(0), "Modelo Equipamento"]
    .value_counts(dropna=False)
    .head(30)
)

print("\n--- MODELOS E QUANTIDADE DE ASSENTOS ---")

resumo_modelos_assentos = (
    pd.DataFrame({
        "Modelo Equipamento": df["Modelo Equipamento"],
        "Número de Assentos": assentos
    })
    .groupby("Modelo Equipamento", dropna=False)["Número de Assentos"]
    .agg(["count", "min", "median", "max"])
    .sort_values("count", ascending=False)
)

print(resumo_modelos_assentos.head(30))

print("\n--- ZERO ASSENTOS POR TIPO DE LINHA ---")
print(
    df.loc[assentos.eq(0), "Código Tipo Linha"]
    .value_counts(dropna=False)
)

######### 2.28 — Valores únicos das principais variáveis categóricas #########

print("\n--- VALORES DAS PRINCIPAIS VARIÁVEIS CATEGÓRICAS ---")

colunas_categoricas = [
    "Situação Voo",
    "Situação Partida",
    "Situação Chegada",
    "Sigla ICAO Empresa Aérea"
]

for coluna in colunas_categoricas:
    print(f"\nColuna: {coluna}")
    print(
        "Quantidade de valores distintos (sem nulos):",
        df[coluna].nunique(dropna=True)
    )
    print(df[coluna].value_counts(dropna=False).head(30))

######### ETAPA 3 — QUALIDADE DOS DADOS #########
# Avaliação da base original, antes do tratamento
# Identifica problemas e limitações sem alterar os dados

# FUNÇÕES AUXILIARES

def percentual_base(quantidade):
    """Calcula o percentual em relação à base original."""
    return quantidade / len(df) * 100 if len(df) > 0 else 0


def percentual_mascara(mascara):
    """Calcula o percentual de registros selecionados na base."""
    return percentual_base(int(mascara.sum()))


def texto_limpo(coluna):
    """Padroniza temporariamente textos para fins de análise."""
    return df[coluna].astype("string").str.strip()


def mascara_ausente(coluna):
    """Identifica NaN, textos vazios e textos compostos por espaços."""
    valores = texto_limpo(coluna)

    return (
        df[coluna].isna()
        | valores.eq("").fillna(False)
    )


def normalizar_categoria(serie):
    """Normaliza categorias sem modificar a coluna original."""
    return (
        serie.astype("string")
        .str.strip()
        .str.upper()
    )


# Total de registros usado como referência em toda a etapa.
total_registros = len(df)

######### 3.1 — COMPLETUDE DOS DADOS #########
# Considera ausentes os NaN e os textos vazios

colunas_relevantes = [
    "Código Tipo Linha",
    "Situação Voo",
    "Situação Partida",
    "Situação Chegada",
    "Partida Prevista",
    "Partida Real",
    "Chegada Prevista",
    "Chegada Real",
    "Número de Assentos",
    "Sigla ICAO Empresa Aérea"
]

ausencias = {
    coluna: int(mascara_ausente(coluna).sum())
    for coluna in colunas_relevantes
}

serie_ausencias = pd.Series(ausencias, dtype="int64")

completude = pd.DataFrame({
    "Ausentes": serie_ausencias,
    "Presentes": total_registros - serie_ausencias,
    "Percentual ausente (%)": (
        serie_ausencias / total_registros * 100
    ).round(3) if total_registros > 0 else 0,
    "Percentual preenchido (%)": (
        (total_registros - serie_ausencias)
        / total_registros * 100
    ).round(3) if total_registros > 0 else 0
})

print("\n--- 3.1 COMPLETUDE DOS DADOS ---")
print(
    completude.sort_values(
        "Percentual ausente (%)",
        ascending=False
    ).to_string()
)

######### 3.2 — UNICIDADE #########
# Diferencia cópias excedentes dos registros envolvidos

total_duplicados = int(df.duplicated().sum())
total_envolvidos = int(df.duplicated(keep=False).sum())
total_unicos = total_registros - total_duplicados

percentual_duplicados = percentual_base(total_duplicados)

print("\n--- 3.2 UNICIDADE DOS DADOS ---")
print(f"Total de registros: {total_registros:,}")
print(
    "Registros após desconsiderar cópias excedentes: "
    f"{total_unicos:,}"
)
print(f"Registros envolvidos em duplicidades: {total_envolvidos:,}")
print(f"Cópias excedentes: {total_duplicados:,}")
print(
    f"Percentual de cópias excedentes: "
    f"{percentual_duplicados:.6f}%"
)

duplicados_exemplo = df.loc[
    df.duplicated(keep=False)
].copy()

if not duplicados_exemplo.empty:

    colunas_ordenacao = [
        coluna
        for coluna in [
            "Sigla ICAO Empresa Aérea",
            "Número Voo",
            "Referência"
        ]
        if coluna in df.columns
    ]

    if colunas_ordenacao:
        duplicados_exemplo = duplicados_exemplo.sort_values(
            colunas_ordenacao
        )

    print("\nExemplos de registros duplicados:")
    print(
        duplicados_exemplo.head(10).to_string(index=False)
    )

else:
    print("\nNenhuma duplicidade exata identificada.")

######### 3.3 — VALIDADE DOS HORÁRIOS #########

colunas_horarios = [
    "Partida Prevista",
    "Partida Real",
    "Chegada Prevista",
    "Chegada Real"
]

horarios_dt = {}
resultados_validade = {}

for coluna in colunas_horarios:

    original = texto_limpo(coluna)

    mascara_nan = df[coluna].isna()

    mascara_vazio = (
        df[coluna].notna()
        & original.eq("").fillna(False)
    )

    mascara_preenchido = (
        df[coluna].notna()
        & original.ne("").fillna(False)
    )

    convertido = pd.to_datetime(
        original,
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    horarios_dt[coluna] = convertido

    mascara_invalido = (
        mascara_preenchido
        & convertido.isna()
    ).fillna(False)

    quantidade_preenchidos = int(
        mascara_preenchido.sum()
    )

    quantidade_invalidos = int(
        mascara_invalido.sum()
    )

    resultados_validade[coluna] = {
        "Ausentes NaN": int(mascara_nan.sum()),
        "Textos vazios": int(mascara_vazio.sum()),
        "Preenchidos inválidos": quantidade_invalidos,
        "Horários válidos": int(convertido.notna().sum()),
        "Máscara inválidos": mascara_invalido,
        "Percentual inválido na base (%)": round(
            percentual_base(quantidade_invalidos),
            3
        ),
        "Percentual inválido entre preenchidos (%)": round(
            quantidade_invalidos / quantidade_preenchidos * 100,
            3
        ) if quantidade_preenchidos > 0 else 0
    }

validade_horarios = pd.DataFrame([
    {
        "Variável": coluna,
        "Ausentes NaN": resultados_validade[coluna]["Ausentes NaN"],
        "Textos vazios": resultados_validade[coluna]["Textos vazios"],
        "Preenchidos inválidos": resultados_validade[coluna]["Preenchidos inválidos"],
        "Horários válidos": resultados_validade[coluna]["Horários válidos"],
        "Percentual inválido (%)": resultados_validade[coluna]["Percentual inválido na base (%)"]
    }
    for coluna in colunas_horarios
])

print("\n--- 3.3 VALIDADE DOS HORÁRIOS ---")
print(validade_horarios.to_string(index=False))

print(
    "\nObservação: a conversão dos horários é auxiliar. "
    "As colunas originais de df permanecem inalteradas."
)

######### 3.4 — VALIDAÇÃO DAS CATEGORIAS #########
# Verifica ausências, valores distintos e categorias inesperadas.

colunas_situacao = [
    "Situação Voo",
    "Situação Partida",
    "Situação Chegada",
    "Código Tipo Linha"
]

print("\n--- 3.4.1 DISTRIBUIÇÃO DAS CATEGORIAS ---")

for coluna in colunas_situacao:

    print(f"\nDistribuição: {coluna}")

    categorias = (
        df[coluna]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    print(
        categorias.value_counts(dropna=False).to_string()
    )

######### 3.4.2 — VALORES NÃO RECONHECIDOS PARA A SITUAÇÃO DO VOO #########

categorias_esperadas_voo = {
    "REALIZADO",
    "CANCELADO"
}

situacao_voo = normalizar_categoria(
    df["Situação Voo"]
)

mascara_situacao_ausente = mascara_ausente(
    "Situação Voo"
)

mascara_situacao_desconhecida = (
    ~mascara_situacao_ausente
    & ~situacao_voo.isin(categorias_esperadas_voo)
).fillna(False)

print("\n--- 3.4.2 VALORES NÃO RECONHECIDOS — SITUAÇÃO DO VOO ---")
print(
    "Situações de voo ausentes ou vazias:",
    int(mascara_situacao_ausente.sum())
)
print(
    "Situações de voo fora das categorias esperadas:",
    int(mascara_situacao_desconhecida.sum())
)

if mascara_situacao_desconhecida.any():

    print("\nValores de situação do voo a investigar:")

    print(
        df.loc[
            mascara_situacao_desconhecida,
            "Situação Voo"
        ].value_counts(dropna=False).to_string()
    )

######### 3.4.3 — CATEGORIAS DE SITUAÇÃO DE PARTIDA E CHEGADA #########

for coluna in ["Situação Partida", "Situação Chegada"]:

    valores = normalizar_categoria(df[coluna])
    ausentes = mascara_ausente(coluna)

    categorias_observadas = sorted(
        valores.loc[~ausentes].dropna().unique().tolist()
    )

    print(f"\nCategorias preenchidas em {coluna}:")
    print(categorias_observadas)

    print(
        f"Registros ausentes ou vazios em {coluna}:",
        int(ausentes.sum())
    )

######### 3.4.4 — VALIDAÇÃO DOS CÓDIGOS DE TIPO DE LINHA #########

categorias_esperadas_linha = {
    "N": "Nacional",
    "I": "Internacional",
    "G": "Geral",
    "C": "Cargueiro"
}

tipo_linha = normalizar_categoria(
    df["Código Tipo Linha"]
)

tipo_linha_ausente = mascara_ausente(
    "Código Tipo Linha"
)

mascara_tipo_linha_nao_reconhecido = (
    ~tipo_linha_ausente
    & ~tipo_linha.isin(categorias_esperadas_linha.keys())
).fillna(False)

print("\n--- 3.4.4 VALIDAÇÃO DOS CÓDIGOS DE TIPO DE LINHA ---")

print("\nDistribuição dos códigos:")
print(
    tipo_linha
    .value_counts(dropna=False)
    .rename_axis("Código")
    .to_frame("Quantidade")
    .to_string()
)

print("\nQuantidade por categoria conhecida:")

for codigo, descricao in categorias_esperadas_linha.items():

    mascara_codigo = tipo_linha.eq(codigo).fillna(False)
    quantidade = int(mascara_codigo.sum())

    print(
        f"{codigo} — {descricao}: {quantidade:,} "
        f"({percentual_base(quantidade):.3f}%)"
    )

print(
    "\nCódigos ausentes ou vazios:",
    int(tipo_linha_ausente.sum())
)

print(
    "Códigos preenchidos não reconhecidos:",
    int(mascara_tipo_linha_nao_reconhecido.sum())
)

if mascara_tipo_linha_nao_reconhecido.any():

    print("\nValores de tipo de linha a investigar:")

    print(
        tipo_linha.loc[
            mascara_tipo_linha_nao_reconhecido
        ].value_counts(dropna=False).to_string()
    )

######### 3.5 — CONSISTÊNCIA OPERACIONAL #########
# Verifica a compatibilidade entre a situação do voo e o preenchimento dos horários reais.

mascara_realizado = situacao_voo.eq("REALIZADO").fillna(False)
mascara_cancelado = situacao_voo.eq("CANCELADO").fillna(False)

partida_real_dt = horarios_dt["Partida Real"]
chegada_real_dt = horarios_dt["Chegada Real"]

verificacoes = {
    "Realizados sem partida real válida": (
        mascara_realizado & partida_real_dt.isna()
    ),
    "Realizados sem chegada real válida": (
        mascara_realizado & chegada_real_dt.isna()
    ),
    "Cancelados com partida real preenchida e válida": (
        mascara_cancelado & partida_real_dt.notna()
    ),
    "Cancelados com chegada real preenchida e válida": (
        mascara_cancelado & chegada_real_dt.notna()
    ),
    "Registros sem situação do voo": mascara_situacao_ausente,
    "Registros com situação não reconhecida": mascara_situacao_desconhecida
}

print("\n--- 3.5 CONSISTÊNCIA OPERACIONAL ---")

for descricao, condicao in verificacoes.items():
    quantidade = int(condicao.sum())

    print(
        f"{descricao}: {quantidade:,} "
        f"({percentual_base(quantidade):.3f}%)"
    )

######### 3.6 — VALIDADE CRONOLÓGICA #########
# Compara horários somente quando ambos estão disponíveis

partida_prevista_dt = horarios_dt["Partida Prevista"]
chegada_prevista_dt = horarios_dt["Chegada Prevista"]

mascara_cronologia_real = (
    partida_real_dt.notna()
    & chegada_real_dt.notna()
)

mascara_cronologia_prevista = (
    partida_prevista_dt.notna()
    & chegada_prevista_dt.notna()
)

chegada_antes_partida = (
    mascara_cronologia_real
    & chegada_real_dt.lt(partida_real_dt)
).fillna(False)

chegada_prevista_antes_partida = (
    mascara_cronologia_prevista
    & chegada_prevista_dt.lt(partida_prevista_dt)
).fillna(False)

print("\n--- 3.6 VALIDADE CRONOLÓGICA ---")

print(
    "Registros com partida e chegada reais disponíveis:",
    int(mascara_cronologia_real.sum())
)

print(
    "Chegada real anterior à partida real:",
    int(chegada_antes_partida.sum())
)

print(
    "Percentual de cronologia real invertida entre avaliáveis:",
    round(
        int(chegada_antes_partida.sum())
        / int(mascara_cronologia_real.sum()) * 100,
        3
    ) if mascara_cronologia_real.sum() > 0 else 0,
    "%"
)

print(
    "Registros não avaliáveis para cronologia real:",
    int((~mascara_cronologia_real).sum())
)

print(
    "\nRegistros com partida e chegada previstas disponíveis:",
    int(mascara_cronologia_prevista.sum())
)

print(
    "Chegada prevista anterior à partida prevista:",
    int(chegada_prevista_antes_partida.sum())
)

print(
    "Percentual de cronologia prevista invertida entre avaliáveis:",
    round(
        int(chegada_prevista_antes_partida.sum())
        / int(mascara_cronologia_prevista.sum()) * 100,
        3
    ) if mascara_cronologia_prevista.sum() > 0 else 0,
    "%"
)

print(
    "Registros não avaliáveis para cronologia prevista:",
    int((~mascara_cronologia_prevista).sum())
)

print(
    "\nObservação: uma chegada anterior à partida pode decorrer "
    "da diferença de fusos horários entre os aeroportos. "
    "Esses casos devem ser investigados antes de serem classificados "
    "como erros."
)

######### 3.7 — QUALIDADE DO NÚMERO DE ASSENTOS #########

assentos_original = texto_limpo("Número de Assentos")

assentos = pd.to_numeric(
    assentos_original,
    errors="coerce"
)

assentos_zero = assentos.eq(0).fillna(False)
assentos_negativos = assentos.lt(0).fillna(False)
assentos_positivos = assentos.gt(0).fillna(False)

mascara_assentos_vazios = mascara_ausente(
    "Número de Assentos"
)

mascara_assentos_nao_numericos = (
    ~mascara_assentos_vazios
    & assentos.isna()
).fillna(False)

assentos_fracionarios = (
    assentos.notna()
    & assentos.mod(1).ne(0)
).fillna(False)

print("\n--- 3.7 QUALIDADE DO NÚMERO DE ASSENTOS ---")
print(f"Total de registros: {total_registros:,}")
print(f"Registros com zero assentos: {int(assentos_zero.sum()):,}")
print(
    f"Percentual com zero assentos: "
    f"{percentual_mascara(assentos_zero):.3f}%"
)
print(
    f"Registros com assentos negativos: "
    f"{int(assentos_negativos.sum()):,}"
)
print(
    f"Registros com assentos positivos: "
    f"{int(assentos_positivos.sum()):,}"
)
print(
    f"Valores ausentes ou vazios: "
    f"{int(mascara_assentos_vazios.sum()):,}"
)
print(
    "Valores preenchidos não numéricos:",
    int(mascara_assentos_nao_numericos.sum())
)
print(
    "Valores fracionários que precisam de verificação:",
    int(assentos_fracionarios.sum())
)

assentos_array = assentos.to_numpy(dtype=float)

classificacao_assentos = np.select(
    [
        np.isnan(assentos_array),
        assentos_array < 0,
        assentos_array == 0,
        assentos_array > 0
    ],
    [
        "Ausente ou não numérico",
        "Valor negativo",
        "Zero assentos",
        "Valor positivo"
    ],
    default="Não classificado"
)

resumo_assentos = pd.Series(
    classificacao_assentos,
    name="Classificação"
).value_counts(dropna=False)

print("\n--- CLASSIFICAÇÃO DOS VALORES DE ASSENTOS ---")
print(resumo_assentos.to_string())

print("\n--- PERCENTUAL POR CLASSIFICAÇÃO ---")
if total_registros > 0:
    print(
        (resumo_assentos / total_registros * 100)
        .round(3)
        .to_string()
    )
else:
    print("Base sem registros.")

print("\n--- ZERO ASSENTOS POR MODELO DE EQUIPAMENTO ---")
print(
    df.loc[assentos_zero]
    .groupby("Modelo Equipamento", dropna=False)
    .size()
    .sort_values(ascending=False)
    .rename("Quantidade")
    .head(20)
    .to_string()
)

print("\n--- ZERO ASSENTOS POR TIPO DE LINHA ---")
print(
    df.loc[assentos_zero]
    .groupby("Código Tipo Linha", dropna=False)
    .size()
    .sort_values(ascending=False)
    .rename("Quantidade")
    .to_string()
)

if assentos_negativos.any():

    print("\n--- REGISTROS COM ASSENTOS NEGATIVOS ---")

    colunas_exibicao = [
        "Empresa Aérea",
        "Número Voo",
        "Modelo Equipamento",
        "Número de Assentos",
        "Código Tipo Linha"
    ]

    print(
        df.loc[assentos_negativos, colunas_exibicao]
        .head(20)
        .to_string(index=False)
    )

if mascara_assentos_nao_numericos.any():

    print("\n--- EXEMPLOS DE ASSENTOS PREENCHIDOS NÃO NUMÉRICOS ---")

    print(
        df.loc[
            mascara_assentos_nao_numericos,
            [
                "Empresa Aérea",
                "Número Voo",
                "Modelo Equipamento",
                "Número de Assentos",
                "Código Tipo Linha"
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

######### 3.8 — COMPLETUDE CONJUNTA DOS HORÁRIOS PREVISTOS #########

partida_prevista_ausente = mascara_ausente("Partida Prevista")
chegada_prevista_ausente = mascara_ausente("Chegada Prevista")

mascara_ambos_previstos_ausentes = (
    partida_prevista_ausente
    & chegada_prevista_ausente
)

mascara_apenas_partida_ausente = (
    partida_prevista_ausente
    & ~chegada_prevista_ausente
)

mascara_apenas_chegada_ausente = (
    ~partida_prevista_ausente
    & chegada_prevista_ausente
)

print("\n--- 3.8 COMPLETUDE CONJUNTA DOS HORÁRIOS PREVISTOS ---")

for descricao, mascara in [
    ("Ambos os horários ausentes", mascara_ambos_previstos_ausentes),
    ("Somente partida prevista ausente", mascara_apenas_partida_ausente),
    ("Somente chegada prevista ausente", mascara_apenas_chegada_ausente)
]:
    quantidade = int(mascara.sum())

    print(
        f"{descricao}: {quantidade:,} "
        f"({percentual_base(quantidade):.3f}%)"
    )

######### 3.9 — CONSISTÊNCIA ENTRE CÓDIGO E NOME DA EMPRESA AÉREA #########
# Verifica relações inconsistentes entre os dois campos

empresas = df[
    [
        "Sigla ICAO Empresa Aérea",
        "Empresa Aérea"
    ]
].copy()

empresas.columns = ["Código", "Empresa"]

empresas["Código"] = (
    empresas["Código"]
    .astype("string")
    .str.strip()
    .str.upper()
)

empresas["Empresa"] = (
    empresas["Empresa"]
    .astype("string")
    .str.strip()
)

empresas = empresas.replace("", pd.NA).dropna(
    subset=["Código", "Empresa"]
)

empresas_por_codigo = (
    empresas.groupby("Código")
    .agg(
        Quantidade_nomes=("Empresa", "nunique"),
        Nomes=("Empresa", lambda x: sorted(x.unique().tolist())),
        Quantidade_registros=("Empresa", "size")
    )
    .reset_index()
)

divergencias_empresas_codigo = empresas_por_codigo[
    empresas_por_codigo["Quantidade_nomes"] > 1
].copy()

print("\n--- CÓDIGOS DE EMPRESA ASSOCIADOS A MAIS DE UM NOME ---")
print(divergencias_empresas_codigo.to_string(index=False))

empresas_por_nome = (
    empresas.groupby("Empresa")
    .agg(
        Quantidade_codigos=("Código", "nunique"),
        Códigos=("Código", lambda x: sorted(x.unique().tolist())),
        Quantidade_registros=("Código", "size")
    )
    .reset_index()
)

divergencias_empresas_nome = empresas_por_nome[
    empresas_por_nome["Quantidade_codigos"] > 1
].copy()

print("\n--- NOMES DE EMPRESA ASSOCIADOS A MAIS DE UM CÓDIGO ---")
print(divergencias_empresas_nome.to_string(index=False))

resumo_consistencia_empresas = pd.DataFrame({
    "Verificação": [
        "Códigos de empresa com múltiplos nomes",
        "Nomes de empresa com múltiplos códigos"
    ],
    "Quantidade de casos": [
        len(divergencias_empresas_codigo),
        len(divergencias_empresas_nome)
    ],
    "Registros envolvidos": [
        int(divergencias_empresas_codigo["Quantidade_registros"].sum()),
        int(divergencias_empresas_nome["Quantidade_registros"].sum())
    ]
})

print("\n--- RESUMO DA CONSISTÊNCIA DAS EMPRESAS AÉREAS ---")
print(resumo_consistencia_empresas.to_string(index=False))

######### 3.9.1 — COMPLETUDE DA IDENTIFICAÇÃO DA EMPRESA AÉREA #########

codigo_empresa_ausente = mascara_ausente(
    "Sigla ICAO Empresa Aérea"
)

nome_empresa_ausente = mascara_ausente(
    "Empresa Aérea"
)

codigo_e_nome_ausentes = (
    codigo_empresa_ausente
    & nome_empresa_ausente
)

somente_codigo_ausente = (
    codigo_empresa_ausente
    & ~nome_empresa_ausente
)

somente_nome_ausente = (
    ~codigo_empresa_ausente
    & nome_empresa_ausente
)

print("\n--- 3.9.1 COMPLETUDE DA IDENTIFICAÇÃO DA EMPRESA ---")

for descricao, mascara in [
    ("Código e nome ausentes", codigo_e_nome_ausentes),
    ("Somente código ausente", somente_codigo_ausente),
    ("Somente nome ausente", somente_nome_ausente)
]:
    quantidade = int(mascara.sum())

    print(
        f"{descricao}: {quantidade:,} "
        f"({percentual_base(quantidade):.3f}%)"
    )

######### 3.10 — VALIDAÇÃO DA LÓGICA DOS ATRASOS #########
# Calcula diferenças entre horários reais e previstos.
# Diferença positiva = atraso.
# Diferença negativa = antecipação.
# Diferença zero = horário exato.

def classificar_diferenca_minutos(minutos):

    if pd.isna(minutos):
        return "Não avaliável"

    if minutos < 0:
        return "Antecipado"

    if minutos == 0:
        return "Pontual (0 min)"

    if minutos <= 29:
        return "Atraso de 1 a 29 min"

    if minutos <= 60:
        return "Atraso de 30 a 60 min"

    if minutos <= 120:
        return "Atraso de 61 a 120 min"

    if minutos <= 240:
        return "Atraso de 121 a 240 min"

    return "Atraso superior a 240 min"

# Diferenças calculadas apenas quando os dois horários existem.
diferenca_partida_min = (
    (partida_real_dt - partida_prevista_dt)
    .dt.total_seconds() / 60
)

diferenca_chegada_min = (
    (chegada_real_dt - chegada_prevista_dt)
    .dt.total_seconds() / 60
)

situacao_partida_original = normalizar_categoria(
    df["Situação Partida"]
)

situacao_chegada_original = normalizar_categoria(
    df["Situação Chegada"]
)

faixa_atraso_partida = diferenca_partida_min.map(
    classificar_diferenca_minutos
)

faixa_atraso_chegada = diferenca_chegada_min.map(
    classificar_diferenca_minutos
)

mascara_partida_avaliavel = (
    mascara_realizado
    & partida_prevista_dt.notna()
    & partida_real_dt.notna()
)

mascara_chegada_avaliavel = (
    mascara_realizado
    & chegada_prevista_dt.notna()
    & chegada_real_dt.notna()
)

mascara_atraso_partida_1_29 = (
    mascara_partida_avaliavel
    & diferenca_partida_min.gt(0)
    & diferenca_partida_min.le(29)
).fillna(False)

mascara_atraso_chegada_1_29 = (
    mascara_chegada_avaliavel
    & diferenca_chegada_min.gt(0)
    & diferenca_chegada_min.le(29)
).fillna(False)

print("\n--- 3.10.1 DISPONIBILIDADE PARA CALCULAR ATRASOS ---")
print(
    "Voos realizados avaliáveis para atraso na partida:",
    int(mascara_partida_avaliavel.sum())
)
print(
    "Voos realizados avaliáveis para atraso na chegada:",
    int(mascara_chegada_avaliavel.sum())
)
print(
    "Voos não avaliáveis para atraso na partida:",
    int((mascara_realizado & ~mascara_partida_avaliavel).sum())
)
print(
    "Voos não avaliáveis para atraso na chegada:",
    int((mascara_realizado & ~mascara_chegada_avaliavel).sum())
)

print("\n--- 3.10.2 FAIXAS DE DIFERENÇA — PARTIDA ---")
print(
    faixa_atraso_partida.loc[mascara_partida_avaliavel]
    .value_counts()
    .reindex(
        [
            "Antecipado",
            "Pontual (0 min)",
            "Atraso de 1 a 29 min",
            "Atraso de 30 a 60 min",
            "Atraso de 61 a 120 min",
            "Atraso de 121 a 240 min",
            "Atraso superior a 240 min"
        ],
        fill_value=0
    )
    .rename_axis("Faixa calculada")
    .to_frame("Quantidade")
    .to_string()
)

print("\n--- 3.10.3 FAIXAS DE DIFERENÇA — CHEGADA ---")
print(
    faixa_atraso_chegada.loc[mascara_chegada_avaliavel]
    .value_counts()
    .reindex(
        [
            "Antecipado",
            "Pontual (0 min)",
            "Atraso de 1 a 29 min",
            "Atraso de 30 a 60 min",
            "Atraso de 61 a 120 min",
            "Atraso de 121 a 240 min",
            "Atraso superior a 240 min"
        ],
        fill_value=0
    )
    .rename_axis("Faixa calculada")
    .to_frame("Quantidade")
    .to_string()
)

print("\n--- 3.10.4 ATRASOS DE 1 A 29 MINUTOS ---")
print(
    "Partidas com atraso de 1 a 29 minutos:",
    int(mascara_atraso_partida_1_29.sum())
)
print(
    "Percentual da base:",
    f"{percentual_mascara(mascara_atraso_partida_1_29):.3f}%"
)
print(
    "Percentual entre partidas avaliáveis:",
    (
        f"{int(mascara_atraso_partida_1_29.sum()) / int(mascara_partida_avaliavel.sum()) * 100:.3f}%"
        if mascara_partida_avaliavel.sum() > 0
        else "Não calculável"
    )
)

print(
    "\nChegadas com atraso de 1 a 29 minutos:",
    int(mascara_atraso_chegada_1_29.sum())
)
print(
    "Percentual da base:",
    f"{percentual_mascara(mascara_atraso_chegada_1_29):.3f}%"
)
print(
    "Percentual entre chegadas avaliáveis:",
    (
        f"{int(mascara_atraso_chegada_1_29.sum()) / int(mascara_chegada_avaliavel.sum()) * 100:.3f}%"
        if mascara_chegada_avaliavel.sum() > 0
        else "Não calculável"
    )
)

print("\n--- 3.10.5 DIFERENÇA CALCULADA X SITUAÇÃO DA PARTIDA ---")
print(
    pd.crosstab(
        faixa_atraso_partida.loc[mascara_partida_avaliavel],
        situacao_partida_original.loc[mascara_partida_avaliavel],
        margins=True,
        dropna=False
    ).to_string()
)

print("\n--- 3.10.6 DIFERENÇA CALCULADA X SITUAÇÃO DA CHEGADA ---")
print(
    pd.crosstab(
        faixa_atraso_chegada.loc[mascara_chegada_avaliavel],
        situacao_chegada_original.loc[mascara_chegada_avaliavel],
        margins=True,
        dropna=False
    ).to_string()
)

######### 3.10.7 — QUANTIFICAÇÃO DAS DIVERGÊNCIAS DE CLASSIFICAÇÃO #########

def resumir_divergencias_atraso(
    diferenca_minutos,
    situacao_original,
    mascara_avaliavel,
    coluna_situacao,
    nome_operacao
):

    diferenca = diferenca_minutos.loc[mascara_avaliavel]
    situacao = situacao_original.loc[mascara_avaliavel]

    mascara_pontual_com_diferenca = (
        diferenca.gt(0)
        & situacao.eq("PONTUAL")
    ).fillna(False)

    mascara_pontual_com_atraso_30_mais = (
        diferenca.ge(30)
        & situacao.eq("PONTUAL")
    ).fillna(False)

    mascara_categoria_ausente = mascara_ausente(
        coluna_situacao
    ).loc[mascara_avaliavel]

    mascara_atraso_com_categoria_ausente = (
        diferenca.gt(0)
        & mascara_categoria_ausente
    ).fillna(False)

    total_avaliavel = int(mascara_avaliavel.sum())

    resultados = [
        (
            "Atraso calculado positivo com categoria PONTUAL",
            int(mascara_pontual_com_diferenca.sum())
        ),
        (
            "Atraso calculado de 30 min ou mais com categoria PONTUAL",
            int(mascara_pontual_com_atraso_30_mais.sum())
        ),
        (
            "Atraso calculado positivo com categoria ausente",
            int(mascara_atraso_com_categoria_ausente.sum())
        )
    ]

    resumo = pd.DataFrame(
        resultados,
        columns=["Verificação", "Quantidade"]
    )

    resumo["Percentual dos avaliáveis (%)"] = (
        resumo["Quantidade"] / total_avaliavel * 100
    ).round(3) if total_avaliavel > 0 else 0

    print(
        f"\n--- 3.10.7 DIVERGÊNCIAS DIAGNÓSTICAS — "
        f"{nome_operacao.upper()} ---"
    )
    print(resumo.to_string(index=False))

    return {
        "atraso_positivo_pontual": int(
            mascara_pontual_com_diferenca.sum()
        ),
        "atraso_30_mais_pontual": int(
            mascara_pontual_com_atraso_30_mais.sum()
        ),
        "atraso_categoria_ausente": int(
            mascara_atraso_com_categoria_ausente.sum()
        )
    }


divergencias_partida = resumir_divergencias_atraso(
    diferenca_partida_min,
    situacao_partida_original,
    mascara_partida_avaliavel,
    "Situação Partida",
    "Partida"
)

divergencias_chegada = resumir_divergencias_atraso(
    diferenca_chegada_min,
    situacao_chegada_original,
    mascara_chegada_avaliavel,
    "Situação Chegada",
    "Chegada"
)

print(
    "\nObservação: a diferença calculada usa a comparação direta "
    "entre horário real e previsto. A categoria 'Pontual' pode "
    "seguir uma tolerância própria. Divergências precisam ser "
    "interpretadas conforme a regra de negócio da fonte."
)

######### 3.11 — REGISTRO CONSOLIDADO DOS ACHADOS #########

partida_prevista_original = texto_limpo("Partida Prevista")
chegada_prevista_original = texto_limpo("Chegada Prevista")

partida_real_original = texto_limpo("Partida Real")
chegada_real_original = texto_limpo("Chegada Real")

partida_real_ausente = mascara_ausente("Partida Real")
chegada_real_ausente = mascara_ausente("Chegada Real")

partida_prevista_invalida = (
    ~partida_prevista_ausente
    & partida_prevista_dt.isna()
).fillna(False)

chegada_prevista_invalida = (
    ~chegada_prevista_ausente
    & chegada_prevista_dt.isna()
).fillna(False)

partida_real_invalida = (
    ~partida_real_ausente
    & partida_real_dt.isna()
).fillna(False)

chegada_real_invalida = (
    ~chegada_real_ausente
    & chegada_real_dt.isna()
).fillna(False)

registros_achados = []


def adicionar_achado(descricao, quantidade, proxima_acao):

    quantidade = int(quantidade)

    registros_achados.append({
        "Achado": descricao,
        "Quantidade": quantidade,
        "Percentual da base (%)": round(
            percentual_base(quantidade), 3
        ),
        "Próxima ação": proxima_acao
    })

# Unicidade
adicionar_achado(
    "Cópias excedentes de registros duplicados",
    total_duplicados,
    "Inspecionar os registros antes de decidir pela remoção"
)

# Completude e categorias
adicionar_achado(
    "Código Tipo Linha ausente ou vazio",
    tipo_linha_ausente.sum(),
    "Investigar a origem dos valores ausentes"
)

adicionar_achado(
    "Código Tipo Linha preenchido não reconhecido",
    mascara_tipo_linha_nao_reconhecido.sum(),
    "Confrontar os valores com a documentação da fonte"
)

adicionar_achado(
    "Situação do voo ausente ou vazia",
    mascara_situacao_ausente.sum(),
    "Verificar os registros sem situação"
)

adicionar_achado(
    "Situação do voo não reconhecida",
    mascara_situacao_desconhecida.sum(),
    "Validar os valores encontrados"
)

adicionar_achado(
    "Situação da partida ausente ou vazia",
    mascara_ausente("Situação Partida").sum(),
    "Investigar os registros sem situação de partida"
)

adicionar_achado(
    "Situação da chegada ausente ou vazia",
    mascara_ausente("Situação Chegada").sum(),
    "Investigar os registros sem situação de chegada"
)

# Horários previstos
adicionar_achado(
    "Partida prevista ausente ou vazia",
    partida_prevista_ausente.sum(),
    "Investigar por empresa e tipo de linha"
)

adicionar_achado(
    "Chegada prevista ausente ou vazia",
    chegada_prevista_ausente.sum(),
    "Investigar por empresa e tipo de linha"
)

adicionar_achado(
    "Ambos os horários previstos ausentes",
    mascara_ambos_previstos_ausentes.sum(),
    "Investigar os registros por empresa e tipo de linha"
)

adicionar_achado(
    "Somente partida prevista ausente",
    mascara_apenas_partida_ausente.sum(),
    "Investigar os registros individualmente"
)

adicionar_achado(
    "Somente chegada prevista ausente",
    mascara_apenas_chegada_ausente.sum(),
    "Investigar os registros individualmente"
)

adicionar_achado(
    "Partida prevista preenchida, mas inválida",
    partida_prevista_invalida.sum(),
    "Inspecionar os valores originais"
)

adicionar_achado(
    "Chegada prevista preenchida, mas inválida",
    chegada_prevista_invalida.sum(),
    "Inspecionar os valores originais"
)

# Horários reais
adicionar_achado(
    "Partida real ausente ou vazia",
    partida_real_ausente.sum(),
    "Investigar por situação do voo e tipo de linha"
)

adicionar_achado(
    "Chegada real ausente ou vazia",
    chegada_real_ausente.sum(),
    "Investigar por situação do voo e tipo de linha"
)

adicionar_achado(
    "Partida real preenchida, mas inválida",
    partida_real_invalida.sum(),
    "Inspecionar os valores originais"
)

adicionar_achado(
    "Chegada real preenchida, mas inválida",
    chegada_real_invalida.sum(),
    "Inspecionar os valores originais"
)

# Consistência operacional
adicionar_achado(
    "Voos realizados sem partida real válida",
    verificacoes["Realizados sem partida real válida"].sum(),
    "Investigar a consistência operacional"
)

adicionar_achado(
    "Voos realizados sem chegada real válida",
    verificacoes["Realizados sem chegada real válida"].sum(),
    "Investigar a consistência operacional"
)

adicionar_achado(
    "Cancelados com partida real válida",
    verificacoes[
        "Cancelados com partida real preenchida e válida"
    ].sum(),
    "Verificar possíveis inconsistências operacionais"
)

adicionar_achado(
    "Cancelados com chegada real válida",
    verificacoes[
        "Cancelados com chegada real preenchida e válida"
    ].sum(),
    "Verificar possíveis inconsistências operacionais"
)

# Cronologia
adicionar_achado(
    "Chegada real anterior à partida real",
    chegada_antes_partida.sum(),
    "Inspecionar horários e referência temporal"
)

adicionar_achado(
    "Chegada prevista anterior à partida prevista",
    chegada_prevista_antes_partida.sum(),
    "Inspecionar horários e referência temporal"
)

# Número de assentos
adicionar_achado(
    "Assentos iguais a zero",
    assentos_zero.sum(),
    "Investigar por modelo, empresa e tipo de linha"
)

adicionar_achado(
    "Assentos negativos",
    assentos_negativos.sum(),
    "Verificar os registros, caso existam"
)

adicionar_achado(
    "Assentos ausentes ou vazios",
    mascara_assentos_vazios.sum(),
    "Investigar os registros sem número de assentos"
)

adicionar_achado(
    "Assentos preenchidos não numéricos",
    mascara_assentos_nao_numericos.sum(),
    "Inspecionar os valores originais"
)

adicionar_achado(
    "Assentos com valores fracionários",
    assentos_fracionarios.sum(),
    "Verificar se os valores são compatíveis com o campo"
)

# Consistência das empresas aéreas
adicionar_achado(
    "Códigos de empresa com múltiplos nomes",
    len(divergencias_empresas_codigo),
    "Investigar códigos associados a mais de um nome"
)

adicionar_achado(
    "Nomes de empresa com múltiplos códigos",
    len(divergencias_empresas_nome),
    "Investigar nomes associados a mais de um código"
)
adicionar_achado(
    "Código e nome da empresa aérea ausentes",
    codigo_e_nome_ausentes.sum(),
    "Investigar a identificação dos registros"
)

adicionar_achado(
    "Somente código da empresa aérea ausente",
    somente_codigo_ausente.sum(),
    "Investigar os registros sem código da empresa"
)

adicionar_achado(
    "Somente nome da empresa aérea ausente",
    somente_nome_ausente.sum(),
    "Investigar os registros sem nome da empresa"
)

# Atrasos de 1 a 29 minutos.
adicionar_achado(
    "Partidas com atraso calculado de 1 a 29 minutos",
    mascara_atraso_partida_1_29.sum(),
    "Confrontar com a regra usada para Situação Partida"
)

adicionar_achado(
    "Chegadas com atraso calculado de 1 a 29 minutos",
    mascara_atraso_chegada_1_29.sum(),
    "Confrontar com a regra usada para Situação Chegada"
)

# Divergências diagnósticas entre cálculos e categorias.
adicionar_achado(
    "Atraso positivo com partida classificada como PONTUAL",
    divergencias_partida["atraso_positivo_pontual"],
    "Confirmar a tolerância e a regra de classificação da fonte"
)

adicionar_achado(
    "Atraso de partida de 30 min ou mais com categoria PONTUAL",
    divergencias_partida["atraso_30_mais_pontual"],
    "Investigar a regra de classificação e os limites das faixas"
)

adicionar_achado(
    "Atraso positivo com chegada classificada como PONTUAL",
    divergencias_chegada["atraso_positivo_pontual"],
    "Confirmar a tolerância e a regra de classificação da fonte"
)

adicionar_achado(
    "Atraso de chegada de 30 min ou mais com categoria PONTUAL",
    divergencias_chegada["atraso_30_mais_pontual"],
    "Investigar a regra de classificação e os limites das faixas"
)

adicionar_achado(
    "Atraso calculado com categoria de partida ausente",
    divergencias_partida["atraso_categoria_ausente"],
    "Investigar a ausência da classificação original"
)

adicionar_achado(
    "Atraso calculado com categoria de chegada ausente",
    divergencias_chegada["atraso_categoria_ausente"],
    "Investigar a ausência da classificação original"
)

######### 3.11.1 — TABELA FINAL DOS ACHADOS #########

achados = pd.DataFrame(registros_achados)

print("\n--- 3.11 REGISTRO CONSOLIDADO DOS ACHADOS ---")
print(achados.to_string(index=False))

######### 3.12 — RESUMO EXECUTIVO DA QUALIDADE DOS DADOS #########

total_horarios_invalidos = sum(
    resultados_validade[coluna]["Preenchidos inválidos"]
    for coluna in colunas_horarios
)

print("\n" + "=" * 70)
print("ETAPA 3 — RESUMO EXECUTIVO DA QUALIDADE DOS DADOS")
print("=" * 70)

print(f"Total de registros analisados: {total_registros:,}")
print(f"Total de colunas: {df.shape[1]:,}")

print(
    f"Cópias excedentes de registros duplicados: "
    f"{total_duplicados:,}"
)

print(
    f"Tipos de linha ausentes ou vazios: "
    f"{int(tipo_linha_ausente.sum()):,}"
)

print(
    f"Tipos de linha preenchidos não reconhecidos: "
    f"{int(mascara_tipo_linha_nao_reconhecido.sum()):,}"
)

print(
    f"Situações de voo ausentes ou vazias: "
    f"{int(mascara_situacao_ausente.sum()):,}"
)

print(
    f"Situações de voo não reconhecidas: "
    f"{int(mascara_situacao_desconhecida.sum()):,}"
)

print(
    f"Registros com ambos os horários previstos ausentes: "
    f"{int(mascara_ambos_previstos_ausentes.sum()):,}"
)

print(
    f"Situações de partida ausentes ou vazias: "
    f"{int(mascara_ausente('Situação Partida').sum()):,}"
)

print(
    f"Situações de chegada ausentes ou vazias: "
    f"{int(mascara_ausente('Situação Chegada').sum()):,}"
)

print(
    f"Registros com zero assentos: "
    f"{int(assentos_zero.sum()):,}"
)

print(
    f"Horários preenchidos inválidos: "
    f"{int(total_horarios_invalidos):,}"
)

print(
    f"Chegadas reais anteriores às partidas reais: "
    f"{int(chegada_antes_partida.sum()):,}"
)

print(
    f"Chegadas previstas anteriores às partidas previstas: "
    f"{int(chegada_prevista_antes_partida.sum()):,}"
)

print(
    f"Voos realizados avaliáveis para atraso na partida: "
    f"{int(mascara_partida_avaliavel.sum()):,}"
)

print(
    f"Voos realizados avaliáveis para atraso na chegada: "
    f"{int(mascara_chegada_avaliavel.sum()):,}"
)

print(
    f"Partidas com atraso calculado de 1 a 29 minutos: "
    f"{int(mascara_atraso_partida_1_29.sum()):,}"
)

print(
    f"Chegadas com atraso calculado de 1 a 29 minutos: "
    f"{int(mascara_atraso_chegada_1_29.sum()):,}"
)

print(
    "\nObservação: os achados representam verificações distintas "
    "e podem se sobrepor. As quantidades não devem ser somadas "
    "para estimar o total de registros problemáticos."
)

print(
    "\nNenhuma linha foi removida, nenhum valor foi imputado "
    "e nenhuma categoria original foi alterada nesta etapa."
)

print("=" * 70)

######### ETAPA 4 — TRATAMENTO, PREPARAÇÃO E BASE ANALÍTICA #########

# Objetivos:
# Preservar a base original.
#Converter os horários para datetime.
# Remover duplicidades exatas.
# Remover Justificativa somente se estiver vazia.
# Validar a base tratada.
# Construir a base analítica.

# REGRA METODOLÓGICA:
# - Voos realizados: referência baseada na Partida Real.
# - Voos cancelados: referência baseada na Partida Prevista.
# - Diferenças de pontualidade: calculadas somente para realizados.

######### ETAPA 4.1 — CONVERSÃO DAS COLUNAS TEMPORAIS #########

print("\n" + "=" * 60)
print("ETAPA 4.1 — CONVERSÃO DAS COLUNAS TEMPORAIS")
print("=" * 60)

df_tratado = df.copy()

colunas_temporais = [
    "Partida Prevista",
    "Partida Real",
    "Chegada Prevista",
    "Chegada Real"
]

colunas_ausentes = [
    coluna for coluna in colunas_temporais
    if coluna not in df_tratado.columns
]

if colunas_ausentes:
    raise KeyError(
        "Colunas temporais não encontradas: "
        + ", ".join(colunas_ausentes)
    )

resultados_conversao = []

for coluna in colunas_temporais:

    original = df_tratado[coluna].copy()

    preenchidos = (
        original.notna()
        & original.astype("string").str.strip().ne("")
    ).fillna(False)

    convertido = pd.to_datetime(
        original,
        format="%d/%m/%Y %H:%M",
        errors="coerce"
    )

    falhas = preenchidos & convertido.isna()

    textos_vazios = (
        original.notna()
        & original.astype("string").str.strip().eq("")
    ).fillna(False)

    resultados_conversao.append({
        "Coluna": coluna,
        "Valores nulos": int(original.isna().sum()),
        "Textos vazios": int(textos_vazios.sum()),
        "Falhas de conversão": int(falhas.sum()),
        "Horários válidos": int(convertido.notna().sum())
    })

    df_tratado[coluna] = convertido

resumo_conversao = pd.DataFrame(resultados_conversao)

print("\n--- RESULTADO DA CONVERSÃO ---")
print(resumo_conversao.to_string(index=False))

print("\n--- TIPOS APÓS A CONVERSÃO ---")
print(df_tratado[colunas_temporais].dtypes)

######### ETAPA 4.2 — TRATAMENTO DE DUPLICIDADES #########

print("\n" + "=" * 60)
print("ETAPA 4.2 — TRATAMENTO DE DUPLICIDADES")
print("=" * 60)

duplicados_antes = int(df_tratado.duplicated().sum())

duplicatas_removidas = df_tratado.loc[
    df_tratado.duplicated(keep="first")
].copy()

print(
    "Cópias excedentes identificadas:",
    f"{duplicados_antes:,}"
)

df_tratado = df_tratado.drop_duplicates().copy()

duplicados_depois = int(df_tratado.duplicated().sum())

registros_removidos = len(df) - len(df_tratado)

print(f"Registros originais: {len(df):,}")
print(f"Registros após remoção: {len(df_tratado):,}")
print(f"Registros removidos: {registros_removidos:,}")
print(f"Duplicidades restantes: {duplicados_depois:,}")

if duplicados_depois == 0:
    print("Validação: nenhuma duplicidade exata restante.")
else:
    print("Atenção: ainda existem duplicidades exatas.")

######### ETAPA 4.3 — AVALIAÇÃO DA COLUNA JUSTIFICATIVA #########

print("\n" + "=" * 60)
print("ETAPA 4.3 — AVALIAÇÃO DA COLUNA JUSTIFICATIVA")
print("=" * 60)

if "Justificativa" in df_tratado.columns:

    justificativa_ausente = (
        df_tratado["Justificativa"].isna()
        | df_tratado["Justificativa"]
            .astype("string")
            .str.strip()
            .eq("")
    ).fillna(False)

    total_ausente = int(justificativa_ausente.sum())
    total_preenchido = len(df_tratado) - total_ausente

    print(f"Valores ausentes ou vazios: {total_ausente:,}")
    print(f"Valores preenchidos: {total_preenchido:,}")

    if total_preenchido == 0:
        df_tratado = df_tratado.drop(
            columns=["Justificativa"]
        )
        print("Coluna Justificativa removida: estava totalmente vazia.")
    else:
        print("Coluna Justificativa mantida: contém valores preenchidos.")

else:
    print("Coluna Justificativa não está presente.")

######### ETAPA 4.4 — VALIDAÇÃO DA BASE TRATADA #########

print("\n" + "=" * 60)
print("ETAPA 4.4 — VALIDAÇÃO DA BASE TRATADA")
print("=" * 60)

print("\n--- DIMENSÕES ---")
print("Base original:", df.shape)
print("Base tratada:", df_tratado.shape)

print("\n--- TIPOS TEMPORAIS ---")
print(df_tratado[colunas_temporais].dtypes)

print("\n--- VALORES AUSENTES NOS HORÁRIOS ---")
print(df_tratado[colunas_temporais].isna().sum())

print("\n--- DUPLICIDADES RESTANTES ---")
print(f"Cópias excedentes: {int(df_tratado.duplicated().sum()):,}")

situacao_voo_tratado = (
    df_tratado["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
)

realizados_mask = situacao_voo_tratado.eq("REALIZADO").fillna(False)
cancelados_mask = situacao_voo_tratado.eq("CANCELADO").fillna(False)

realizados_sem_partida = (
    realizados_mask & df_tratado["Partida Real"].isna()
)

realizados_sem_chegada = (
    realizados_mask & df_tratado["Chegada Real"].isna()
)

cancelados_com_partida_real = (
    cancelados_mask & df_tratado["Partida Real"].notna()
)

cancelados_com_chegada_real = (
    cancelados_mask & df_tratado["Chegada Real"].notna()
)

print("\n--- CONSISTÊNCIA DOS VOOS REALIZADOS ---")
print(
    "Realizados sem partida real válida:",
    f"{int(realizados_sem_partida.sum()):,}"
)
print(
    "Realizados sem chegada real válida:",
    f"{int(realizados_sem_chegada.sum()):,}"
)

print("\n--- CONSISTÊNCIA DOS VOOS CANCELADOS ---")
print(
    "Cancelados com partida real preenchida:",
    f"{int(cancelados_com_partida_real.sum()):,}"
)
print(
    "Cancelados com chegada real preenchida:",
    f"{int(cancelados_com_chegada_real.sum()):,}"
)

######### 4.4.1 — VALIDAR REGISTROS COM OS QUATRO HORÁRIOS #########

horarios_completos_original = pd.DataFrame(
    {
        coluna: pd.to_datetime(
            df[coluna],
            format="%d/%m/%Y %H:%M",
            errors="coerce"
        )
        for coluna in colunas_temporais
    },
    index=df.index
)

situacao_voo_original = (
    df["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
)

realizados_original_mask = (
    situacao_voo_original.eq("REALIZADO").fillna(False)
)

avaliavel_original = (
    realizados_original_mask
    & horarios_completos_original.notna().all(axis=1)
)

avaliavel_tratado = (
    realizados_mask
    & df_tratado[colunas_temporais].notna().all(axis=1)
)

print("\n--- REGISTROS COM OS QUATRO HORÁRIOS VÁLIDOS ---")
print("Base original:", f"{int(avaliavel_original.sum()):,}")
print("Base tratada:", f"{int(avaliavel_tratado.sum()):,}")
print(
    "Diferença entre contagens:",
    f"{int(avaliavel_original.sum() - avaliavel_tratado.sum()):,}"
)

######### 4.4.2 — AVALIAR AS CÓPIAS EXCEDENTES REMOVIDAS #########

if not duplicatas_removidas.empty:

    situacao_duplicatas = (
        duplicatas_removidas["Situação Voo"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    duplicatas_avaliaveis = (
        situacao_duplicatas.eq("REALIZADO")
        & duplicatas_removidas[colunas_temporais].notna().all(axis=1)
    )

    total_duplicatas_avaliaveis = int(duplicatas_avaliaveis.sum())

else:
    total_duplicatas_avaliaveis = 0

print("\n--- CÓPIAS EXCEDENTES REMOVIDAS ---")
print(f"Cópias removidas: {len(duplicatas_removidas):,}")
print(
    "Voos realizados removidos com os quatro horários válidos:",
    f"{total_duplicatas_avaliaveis:,}"
)

######### ETAPA 4.5 — RESUMO DO TRATAMENTO #########

print("\n" + "=" * 60)
print("ETAPA 4.5 — RESUMO DO TRATAMENTO")
print("=" * 60)

print(f"Registros originais: {len(df):,}")
print(f"Registros tratados: {len(df_tratado):,}")
print(f"Colunas originais: {df.shape[1]}")
print(f"Colunas tratadas: {df_tratado.shape[1]}")
print(f"Registros removidos por duplicidade: {registros_removidos:,}")
print(f"Duplicidades restantes: {int(df_tratado.duplicated().sum()):,}")
print(f"Voos realizados: {int(realizados_mask.sum()):,}")
print(f"Voos cancelados: {int(cancelados_mask.sum()):,}")
print(
    "Realizados sem partida real válida:",
    f"{int(realizados_sem_partida.sum()):,}"
)
print(
    "Realizados sem chegada real válida:",
    f"{int(realizados_sem_chegada.sum()):,}"
)

######### ETAPA 4.6 — CONSTRUÇÃO DA BASE ANALÍTICA #########
print("\n" + "=" * 60)
print("ETAPA 4.6 — CONSTRUÇÃO DA BASE ANALÍTICA")
print("=" * 60)

df_analise = df_tratado.copy()

situacao_voo_analise = (
    df_analise["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
)

voos_realizados_mask = (
    situacao_voo_analise.eq("REALIZADO").fillna(False)
)

voos_cancelados_mask = (
    situacao_voo_analise.eq("CANCELADO").fillna(False)
)

######### 4.6.1 — DATA DE REFERÊNCIA PARA ANÁLISES MENSAIS #########
# REALIZADO - usar Partida Real.
# CANCELADO - usar Partida Prevista.
# Não usar Partida Prevista como referência mensal para os
# voos realizados.

df_analise["Data Referência"] = pd.NaT

# Para voos realizados, a referência é a data da operação real.
df_analise.loc[
    voos_realizados_mask,
    "Data Referência"
] = df_analise.loc[
    voos_realizados_mask,
    "Partida Real"
]

# Para voos cancelados, a referência é a data da operação programada.
df_analise.loc[
    voos_cancelados_mask,
    "Data Referência"
] = df_analise.loc[
    voos_cancelados_mask,
    "Partida Prevista"
]

df_analise["Mês Referência"] = (
    df_analise["Data Referência"].dt.month
)

df_analise["Ano Referência"] = (
    df_analise["Data Referência"].dt.year
)

print("\n--- DATA DE REFERÊNCIA ---")
print("Regra para realizados: Partida Real")
print("Regra para cancelados: Partida Prevista")
print(
    "Registros sem data de referência:",
    f"{int(df_analise['Data Referência'].isna().sum()):,}"
)
print(
    "Realizados sem data de referência:",
    f"{int((voos_realizados_mask & df_analise['Data Referência'].isna()).sum()):,}"
)
print(
    "Cancelados sem data de referência:",
    f"{int((voos_cancelados_mask & df_analise['Data Referência'].isna()).sum()):,}"
)

######### 4.6.2 — DIFERENÇAS ENTRE HORÁRIOS PREVISTOS E REAIS #########

df_analise["Diferença Partida (min)"] = np.nan
df_analise["Diferença Chegada (min)"] = np.nan

mascara_partida_calculavel = (
    voos_realizados_mask
    & df_analise["Partida Prevista"].notna()
    & df_analise["Partida Real"].notna()
)

df_analise.loc[
    mascara_partida_calculavel,
    "Diferença Partida (min)"
] = (
    (
        df_analise.loc[mascara_partida_calculavel, "Partida Real"]
        - df_analise.loc[mascara_partida_calculavel, "Partida Prevista"]
    ).dt.total_seconds() / 60
)

mascara_chegada_calculavel = (
    voos_realizados_mask
    & df_analise["Chegada Prevista"].notna()
    & df_analise["Chegada Real"].notna()
)

df_analise.loc[
    mascara_chegada_calculavel,
    "Diferença Chegada (min)"
] = (
    (
        df_analise.loc[mascara_chegada_calculavel, "Chegada Real"]
        - df_analise.loc[mascara_chegada_calculavel, "Chegada Prevista"]
    ).dt.total_seconds() / 60
)

######### 4.6.3 — SINALIZAÇÃO DE DIFERENÇAS EXTREMAS #########

limite_extremo = 1440  # 24 horas

df_analise["Partida com diferença extrema"] = (
    df_analise["Diferença Partida (min)"]
    .abs()
    .gt(limite_extremo)
    .fillna(False)
)

df_analise["Chegada com diferença extrema"] = (
    df_analise["Diferença Chegada (min)"]
    .abs()
    .gt(limite_extremo)
    .fillna(False)
)

######### 4.6.4 — CRIAR VISÃO DOS VOOS REALIZADOS #########

df_realizados = df_analise.loc[
    voos_realizados_mask
].copy()

######### 4.6.5 — VALIDAR A BASE ANALÍTICA #########

print("\n" + "=" * 60)
print("ETAPA 4.6.5 — VALIDAÇÃO DA BASE ANALÍTICA")
print("=" * 60)

print(f"Quantidade de registros: {len(df_analise):,}")
print(f"Quantidade de colunas: {len(df_analise.columns):,}")
print(f"Voos realizados: {len(df_realizados):,}")
print(f"Voos cancelados: {int(voos_cancelados_mask.sum()):,}")

print(
    "Registros sem data de referência:",
    f"{int(df_analise['Data Referência'].isna().sum()):,}"
)

print(
    "Realizados sem diferença calculável na partida:",
    f"{int(df_realizados['Diferença Partida (min)'].isna().sum()):,}"
)

print(
    "Realizados sem diferença calculável na chegada:",
    f"{int(df_realizados['Diferença Chegada (min)'].isna().sum()):,}"
)

print(
    "Partidas com diferença extrema:",
    f"{int(df_analise['Partida com diferença extrema'].sum()):,}"
)

print(
    "Chegadas com diferença extrema:",
    f"{int(df_analise['Chegada com diferença extrema'].sum()):,}"
)

print(
    "Registros extremos em partida ou chegada:",
    f"{int((df_analise['Partida com diferença extrema'] | df_analise['Chegada com diferença extrema']).sum()):,}"
)

######### ETAPA 4.7 — VALIDAÇÃO FINAL DAS BASES #########

print("\n" + "=" * 60)
print("ETAPA 4.7 — VALIDAÇÃO FINAL DAS BASES")
print("=" * 60)

print("Base original (df):", df.shape)
print("Base tratada (df_tratado):", df_tratado.shape)
print("Base analítica (df_analise):", df_analise.shape)

print("\n--- VERIFICAÇÕES FINAIS ---")

print(
    "Base tratada sem duplicidades:",
    df_tratado.duplicated().sum() == 0
)

print(
    "Colunas temporais convertidas:",
    all(
        pd.api.types.is_datetime64_any_dtype(df_tratado[coluna])
        for coluna in colunas_temporais
    )
)

print(
    "Base de voos realizados consistente:",
    len(df_realizados) == int(voos_realizados_mask.sum())
)

print(
    "Base original preservada em quantidade de registros:",
    len(df) == len(df_tratado) + registros_removidos
)

# Validar a regra da data de referência.
referencia_realizados_correta = df_analise.loc[
    voos_realizados_mask, "Data Referência"
].equals(
    df_analise.loc[voos_realizados_mask, "Partida Real"]
)

referencia_cancelados_correta = df_analise.loc[
    voos_cancelados_mask, "Data Referência"
].equals(
    df_analise.loc[voos_cancelados_mask, "Partida Prevista"]
)

print(
    "Referência dos realizados baseada na Partida Real:",
    referencia_realizados_correta
)

print(
    "Referência dos cancelados baseada na Partida Prevista:",
    referencia_cancelados_correta
)

print("=" * 60)

######### ETAPA 5 — ANÁLISE EXPLORATÓRIA DOS DADOS #########

# Objetivos:
# 1. Analisar o volume mensal de voos realizados.
# 2. Calcular a taxa mensal de cancelamento por mês de partida prevista.
# 3. Explorar as classificações de partida e chegada da ANAC.
# 5. Analisar diferenças entre horários previstos e reais.
# 6. Investigar diferenças extremas e divergências de classificação.
#
# CRITÉRIOS METODOLÓGICOS:
# - A situação do voo determina se foi realizado ou cancelado.
# - O volume mensal de realizados utiliza a Partida Real.
# - A comparação mensal e a taxa de cancelamento utilizam a Partida Prevista para os registros com situação conhecida.
# - Registros sem Partida Prevista não são atribuídos artificialmente a um mês programado.
# - Diferenças extremas são investigadas, não removidas automaticamente.
# - O tratamento de dados e a remoção de duplicatas foram feitos na Etapa 4.

######### ETAPA 5.0 — PREPARAÇÃO DAS BASES DE ANÁLISE #########

print("\n" + "=" * 60)
print("ETAPA 5 — ANÁLISE EXPLORATÓRIA DOS DADOS")
print("=" * 60)

if "df_analise" not in globals():
    raise NameError("Execute a Etapa 4 para criar df_analise.")

if "df_realizados" not in globals():
    raise NameError("Execute a Etapa 4 para criar df_realizados.")

# Criar cópias para não alterar as bases originais da Etapa 4.
df_eda = df_analise.copy()
df_voos_realizados = df_realizados.copy()

# Limite para investigar diferenças superiores a 24 horas.
limite_extremo = 1440

meses = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]

situacoes_validas = ["REALIZADO", "CANCELADO"]

colunas_temporais = [
    "Partida Prevista",
    "Partida Real",
    "Chegada Prevista",
    "Chegada Real"
]

colunas_diferencas = [
    "Diferença Partida (min)",
    "Diferença Chegada (min)"
]

colunas_obrigatorias = [
    "Situação Voo",
    "Situação Partida",
    "Situação Chegada",
    *colunas_temporais,
    *colunas_diferencas
]

def padronizar_categoria(serie):
    """
    Padroniza categorias conhecidas, preservando outros valores
    encontrados na fonte.
    """
    valores = serie.astype("string").str.strip()
    valores = valores.mask(valores.eq(""), pd.NA)

    mapa = {
        "REALIZADO": "REALIZADO",
        "CANCELADO": "CANCELADO",
        "ANTECIPADO": "Antecipado",
        "PONTUAL": "Pontual",
        "ATRASO 30-60": "Atraso 30-60",
        "ATRASO 60-120": "Atraso 60-120",
        "ATRASO 120-240": "Atraso 120-240",
        "ATRASO > 240": "Atraso > 240"
    }

    normalizados = valores.str.upper().map(mapa)

    return normalizados.fillna(valores)


# Conferir as colunas necessárias na base analítica.
colunas_ausentes = [
    coluna for coluna in colunas_obrigatorias
    if coluna not in df_eda.columns
]

if colunas_ausentes:
    raise KeyError(
        "Colunas ausentes em df_analise: "
        + ", ".join(colunas_ausentes)
    )

# Conferir as colunas necessárias na base de realizados.
colunas_ausentes_realizados = [
    coluna for coluna in colunas_obrigatorias
    if coluna not in df_voos_realizados.columns
]

if colunas_ausentes_realizados:
    raise KeyError(
        "Colunas ausentes em df_realizados: "
        + ", ".join(colunas_ausentes_realizados)
    )

# Conferir os formatos temporais sem converter novamente os dados.
for coluna in colunas_temporais:

    if not pd.api.types.is_datetime64_any_dtype(df_eda[coluna]):
        raise TypeError(
            f"'{coluna}' não está em formato datetime. "
            "Verifique a Etapa 4."
        )

# Padronizar a situação somente nas cópias de análise.
df_eda["Situação Voo Análise"] = (
    padronizar_categoria(df_eda["Situação Voo"])
)

df_voos_realizados["Situação Voo Análise"] = (
    padronizar_categoria(df_voos_realizados["Situação Voo"])
)

mask_realizado = (
    df_eda["Situação Voo Análise"] == "REALIZADO"
)

mask_cancelado = (
    df_eda["Situação Voo Análise"] == "CANCELADO"
)

mask_situacao_valida = (
    df_eda["Situação Voo Análise"].isin(situacoes_validas)
)
if not df_voos_realizados["Situação Voo Análise"].eq("REALIZADO").all():
    raise ValueError(
        "df_realizados contém registros que não estão classificados "
        "como REALIZADO. Verifique a Etapa 4."
    )

# Exibir as dimensões das bases
print("Base analítica original:", df_analise.shape)
print("Base de trabalho da EDA:", df_eda.shape)
print("Base de voos realizados:", df_voos_realizados.shape)

print("\nSituação dos voos na base analítica:")
print(df_eda["Situação Voo Análise"].value_counts(dropna=False))

print("\nPreparação concluída.")


######### ETAPA 5.1 — VOLUME DE VOOS REALIZADOS POR MÊS #########

df_voos_realizados["Mês Partida Real"] = (
    df_voos_realizados["Partida Real"].dt.month
)

voos_por_mes = (
    df_voos_realizados
    .dropna(subset=["Mês Partida Real"])
    .groupby("Mês Partida Real")
    .size()
    .reindex(range(1, 13), fill_value=0)
)

voos_sem_partida_real = int(
    df_voos_realizados["Partida Real"].isna().sum()
)

voos_por_mes.index = meses
voos_por_mes.index.name = "Mês"

print("\nVOLUME DE VOOS REALIZADOS POR MÊS")
print(voos_por_mes.to_string())

participacao_mes = (
    voos_por_mes
    .div(voos_por_mes.sum())
    .mul(100)
    .round(2)
)

print(
    "\nPARTICIPAÇÃO DOS VOOS REALIZADOS NO ANO (%)"
)

print(participacao_mes.to_string())

print(
    "\nVoos realizados contabilizados por mês:",
    f"{int(voos_por_mes.sum()):,}"
)

print(
    "Total de voos realizados na base:",
    f"{len(df_voos_realizados):,}"
)

print(
    "Voos sem partida real válida:",
    f"{voos_sem_partida_real:,}"
)

print(
    "Conferência dos registros:",
    f"{int(voos_por_mes.sum()) + voos_sem_partida_real:,}",
    "de",
    f"{len(df_voos_realizados):,}"
)

assert (
    int(voos_por_mes.sum()) + voos_sem_partida_real
    == len(df_voos_realizados)
), (
    "O volume mensal não fecha com o total de realizados."
)

######### ETAPA 5.2 — COMPARAÇÃO MENSAL E TAXA DE CANCELAMENTO #########

df_eda["Mês Partida Prevista"] = (
    df_eda["Partida Prevista"].dt.month
)

mask_partida_prevista_valida = (
    df_eda["Partida Prevista"].notna()
)

mask_programacao_valida = (
    mask_situacao_valida
    & mask_partida_prevista_valida
)

df_programados = (
    df_eda
    .loc[mask_programacao_valida]
    .copy()
)

resultado_cancelamento = (
    pd.crosstab(
        df_programados["Mês Partida Prevista"],
        df_programados["Situação Voo Análise"]
    )
    .reindex(
        range(1, 13),
        fill_value=0
    )
)

for coluna in situacoes_validas:

    if coluna not in resultado_cancelamento.columns:
        resultado_cancelamento[coluna] = 0

resultado_cancelamento = (
    resultado_cancelamento[
        ["REALIZADO", "CANCELADO"]
    ]
    .copy()
)

resultado_cancelamento["TOTAL PROGRAMADO"] = (
    resultado_cancelamento["REALIZADO"]
    + resultado_cancelamento["CANCELADO"]
)

resultado_cancelamento["TAXA_CANCELAMENTO (%)"] = (
    resultado_cancelamento["CANCELADO"]
    .div(
        resultado_cancelamento["TOTAL PROGRAMADO"]
        .replace(0, np.nan)
    )
    .mul(100)
    .round(2)
)

resultado_cancelamento.index.name = (
    "Mês numérico da partida prevista"
)

registros_sem_partida_prevista = int(
    (~mask_partida_prevista_valida).sum()
)

registros_situacao_invalida = int(
    (~mask_situacao_valida).sum()
)

registros_sem_programacao_avaliavel = int(
    (~mask_programacao_valida).sum()
)

print("\nCOMPARAÇÃO MENSAL E TAXA DE CANCELAMENTO")

print(
    resultado_cancelamento.rename(
        index=lambda mes: meses[int(mes) - 1]
    ).to_string()
)

print(
    "\nTotal de registros incluídos no cálculo:",
    f"{len(df_programados):,}"
)

print(
    "Total de registros na base analítica:",
    f"{len(df_eda):,}"
)

print(
    "Registros sem partida prevista:",
    f"{registros_sem_partida_prevista:,}"
)

print(
    "Registros com situação inválida:",
    f"{registros_situacao_invalida:,}"
)

print(
    "Registros fora do cálculo da taxa:",
    f"{registros_sem_programacao_avaliavel:,}"
)

ranking_cancelamento = (
    resultado_cancelamento
    .sort_values(
        "TAXA_CANCELAMENTO (%)",
        ascending=False
    )
)

print(
    "\nMESES COM MAIOR TAXA DE CANCELAMENTO"
)

print(
    ranking_cancelamento[
        [
            "CANCELADO",
            "TOTAL PROGRAMADO",
            "TAXA_CANCELAMENTO (%)"
        ]
    ]
)

assert (
    int(resultado_cancelamento["CANCELADO"].sum())
    ==
    int(
        (
            mask_cancelado
            & mask_partida_prevista_valida
        ).sum()
    )
), (
    "A contagem de cancelamentos não confere."
)

assert (
    int(resultado_cancelamento["TOTAL PROGRAMADO"].sum())
    ==
    len(df_programados)
), (
    "O total programado não confere."
)

# 5.2.1 — DADOS DA EVOLUÇÃO MENSAL DAS OPERAÇÕES

nomes_meses_programados = [
    meses[int(mes) - 1]
    for mes in resultado_cancelamento.index
]

# 5.2.2 — DADOS DO VOLUME DE OPERAÇÕES POR TIPO DE LINHA

df_tipo_linha = df_tratado[
    [
        "Código Tipo Linha",
        "Situação Voo"
    ]
].copy()

# Padronização
df_tipo_linha["Código Tipo Linha"] = (
    df_tipo_linha["Código Tipo Linha"]
    .astype("string")
    .str.strip()
    .str.upper()
)

df_tipo_linha["Situação Voo"] = (
    df_tipo_linha["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
)

df_tipo_linha["Código Tipo Linha"] = (
    df_tipo_linha["Código Tipo Linha"]
    .replace("", pd.NA)
)

# Categorias
df_tipo_linha["Categoria"] = (
    df_tipo_linha["Código Tipo Linha"]
    .fillna("Ausente")
)

categorias_tipo_linha = {
    "N": "Nacional",
    "I": "Internacional",
    "G": "Geral",
    "C": "Cargueiro",
    "Ausente": "Não classificado"
}

df_tipo_linha["Descrição Categoria"] = (
    df_tipo_linha["Categoria"]
    .map(categorias_tipo_linha)
    .fillna("Outros códigos")
)

# Resumo para o Dashboard
resumo_tipo_linha = (
    df_tipo_linha[
        df_tipo_linha["Situação Voo"].isin(
            [
                "REALIZADO",
                "CANCELADO"
            ]
        )
    ]
    .groupby(
        [
            "Categoria",
            "Descrição Categoria",
            "Situação Voo"
        ],
        dropna=False
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for coluna in ["REALIZADO", "CANCELADO"]:

    if coluna not in resumo_tipo_linha.columns:
        resumo_tipo_linha[coluna] = 0

resumo_tipo_linha["TOTAL"] = (
    resumo_tipo_linha["REALIZADO"]
    + resumo_tipo_linha["CANCELADO"]
)

resumo_tipo_linha = (
    resumo_tipo_linha
    .sort_values(
        "TOTAL",
        ascending=False
    )
)

print(
    "\nVOLUME DE OPERAÇÕES POR TIPO DE LINHA"
)

print(
    resumo_tipo_linha[
        [
            "Categoria",
            "Descrição Categoria",
            "REALIZADO",
            "CANCELADO",
            "TOTAL"
        ]
    ]
    .to_string(index=False)
)

######### 5.2.3 — DADOS DO VOLUME DE OPERAÇÕES POR EMPRESA AÉREA #########

df_empresas = df_tratado[
    [
        "Sigla ICAO Empresa Aérea",
        "Empresa Aérea",
        "Situação Voo"
    ]
].copy()

df_empresas["Situação Voo"] = (
    df_empresas["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
)

resumo_empresas = (
    df_empresas[
        df_empresas["Situação Voo"].isin(
            [
                "REALIZADO",
                "CANCELADO"
            ]
        )
    ]
    .groupby(
        [
            "Sigla ICAO Empresa Aérea",
            "Empresa Aérea",
            "Situação Voo"
        ],
        dropna=False
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for coluna in ["REALIZADO", "CANCELADO"]:

    if coluna not in resumo_empresas.columns:
        resumo_empresas[coluna] = 0

resumo_empresas["TOTAL"] = (
    resumo_empresas["REALIZADO"]
    + resumo_empresas["CANCELADO"]
)

top_empresas = (
    resumo_empresas
    .sort_values(
        "TOTAL",
        ascending=False
    )
    .head(15)
)

print(
    "\nTOP 15 EMPRESAS POR VOLUME DE OPERAÇÕES"
)

print(
    top_empresas[
        [
            "Empresa Aérea",
            "REALIZADO",
            "CANCELADO",
            "TOTAL"
        ]
    ]
    .to_string(index=False)
)

######### 5.2.4 — DADOS DAS EMPRESAS COM MAIORES TAXAS DE CANCELAMENTO #########

mask_programacao_valida = (
    df_eda["Partida Prevista"].notna()
    &
    df_eda["Situação Voo"].isin(
        [
            "REALIZADO",
            "CANCELADO"
        ]
    )
    &
    df_eda["Empresa Aérea"].notna()
    &
    df_eda["Empresa Aérea"]
    .astype("string")
    .str.strip()
    .ne("")
)

df_empresas_cancelamento = (
    df_eda
    .loc[
        mask_programacao_valida
    ]
    .copy()
)

resultado_empresas_cancelamento = pd.crosstab(
    df_empresas_cancelamento["Empresa Aérea"],
    df_empresas_cancelamento["Situação Voo"]
)

for coluna in ["REALIZADO", "CANCELADO"]:

    if coluna not in resultado_empresas_cancelamento.columns:
        resultado_empresas_cancelamento[coluna] = 0

resultado_empresas_cancelamento = (
    resultado_empresas_cancelamento[
        [
            "REALIZADO",
            "CANCELADO"
        ]
    ]
    .copy()
)

resultado_empresas_cancelamento["TOTAL PROGRAMADO"] = (
    resultado_empresas_cancelamento["REALIZADO"]
    +
    resultado_empresas_cancelamento["CANCELADO"]
)

resultado_empresas_cancelamento[
    "TAXA CANCELAMENTO (%)"
] = (
    resultado_empresas_cancelamento["CANCELADO"]
    .div(
        resultado_empresas_cancelamento["TOTAL PROGRAMADO"]
    )
    .mul(100)
)

volume_minimo = 1000

top_10_empresas_cancelamento = (
    resultado_empresas_cancelamento
    .loc[
        lambda x:
        x["TOTAL PROGRAMADO"] >= volume_minimo
    ]
    .sort_values(
        "TAXA CANCELAMENTO (%)",
        ascending=False
    )
    .head(10)
)

print(
    "\n10 COMPANHIAS COM MAIOR TAXA DE CANCELAMENTO"
)

print(
    top_10_empresas_cancelamento.round(2)
)

dados_grafico = (
    top_10_empresas_cancelamento
    .copy()
)

nomes_exibicao = {
    "ARAJET S.A.": "ARAJET",
    "ETHIOPIAN AIRLINES GROUP": "ETHIOPIAN AIRLINES",
    "IBÉRIA LINEAS AEREAS DE ESPAÑA SOCIEDAD ANONIMA OPERADORA":
        "IBÉRIA LINEAS",
    "QATAR AIRWAYS GROUP": "QATAR AIRWAYS",
    "TAMPA CARGO S.A.": "TAMPA CARGO",
    "ABSA - AEROLINHAS BRASILEIRAS S.A.": "ABSA",
    "ITALIA TRANSPORTO AEREO S.P.A.": "ITALIA TRANSPORTO AEREO",
    "TOTAL LINHAS AÉREAS S.A.": "TOTAL LINHAS AÉREAS",
    "ATLAS AIR INC": "ATLAS AIR",
    "ATA - AEROTÁXI ABAETÉ LTDA.": "ATA - AEROTÁXI ABAETÉ"
}

dados_grafico["Empresa Aérea"] = (
    dados_grafico.index.to_series()
    .map(
        lambda nome:
        nomes_exibicao.get(nome, nome)
    )
)

valores_cancelados = (
    dados_grafico["CANCELADO"]
)

maior_valor_cancelado = (
    valores_cancelados.max()
)

if maior_valor_cancelado > 0:

    tamanhos_pontos = (
        18
        + 30
        * valores_cancelados
        / maior_valor_cancelado
    )

else:

    tamanhos_pontos = 18

cores_pontos = [
    "#D97745"
    if taxa >= 30
    else "#245A81"
    for taxa
    in dados_grafico[
        "TAXA CANCELAMENTO (%)"
    ]
]

######### 5.2.5 — DADOS DA EVOLUÇÃO MENSAL DAS TAXAS DE CANCELAMENTO #########

empresas_selecionadas = (
    top_10_empresas_cancelamento.index.tolist()
)

mask_mensal_valida = (

    df_eda["Partida Prevista"].notna()

    &

    df_eda["Situação Voo"].isin(
        [
            "REALIZADO",
            "CANCELADO"
        ]
    )

    &

    df_eda["Empresa Aérea"].isin(
        empresas_selecionadas
    )

)

df_mensal_empresas = (
    df_eda
    .loc[
        mask_mensal_valida
    ]
    .copy()
)

df_mensal_empresas["Mês"] = (
    df_mensal_empresas["Partida Prevista"]
    .dt.month
)

resultado_mensal_empresas = (

    df_mensal_empresas

    .groupby(
        [
            "Empresa Aérea",
            "Mês",
            "Situação Voo"
        ]
    )

    .size()

    .unstack(fill_value=0)

)

for coluna in [
    "REALIZADO",
    "CANCELADO"
]:

    if coluna not in resultado_mensal_empresas.columns:
        resultado_mensal_empresas[coluna] = 0

resultado_mensal_empresas = (
    resultado_mensal_empresas[
        [
            "REALIZADO",
            "CANCELADO"
        ]
    ]
)

resultado_mensal_empresas[
    "TOTAL PROGRAMADO"
] = (

    resultado_mensal_empresas["REALIZADO"]

    +

    resultado_mensal_empresas["CANCELADO"]

)

resultado_mensal_empresas[
    "TAXA CANCELAMENTO (%)"
] = (

    resultado_mensal_empresas["CANCELADO"]

    .div(
        resultado_mensal_empresas[
            "TOTAL PROGRAMADO"
        ].replace(0, np.nan)
    )

    .mul(100)

)

taxas_mensais = (

    resultado_mensal_empresas[
        "TAXA CANCELAMENTO (%)"
    ]

    .unstack("Mês")

    .reindex(
        index=empresas_selecionadas,
        columns=range(1, 13)
    )

)

cancelados_mensais = (

    resultado_mensal_empresas[
        "CANCELADO"
    ]

    .unstack("Mês")

    .reindex(
        index=empresas_selecionadas,
        columns=range(1, 13)
    )

)

programados_mensais = (

    resultado_mensal_empresas[
        "TOTAL PROGRAMADO"
    ]

    .unstack("Mês")

    .reindex(
        index=empresas_selecionadas,
        columns=range(1, 13)
    )

)

nomes_meses = [
    "Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
    "Jul", "Ago", "Set", "Out", "Nov", "Dez"
]

print(
    "\nEVOLUÇÃO MENSAL DAS TAXAS DE CANCELAMENTO (%)"
)

print(
    taxas_mensais
    .rename(
        columns=dict(
            enumerate(
                nomes_meses,
                start=1
            )
        )
    )
    .round(2)
    .to_string()
)

nomes_exibicao = {
    "ARAJET S.A.": "ARAJET",
    "ETHIOPIAN AIRLINES GROUP": "ETHIOPIAN AIRLINES",
    "QATAR AIRWAYS GROUP": "QATAR AIRWAYS",
    "ATLAS AIR INC": "ATLAS AIR",
    "TAMPA CARGO S.A.": "TAMPA CARGO",
    "ABSA - AEROLINHAS BRASILEIRAS S.A.": "ABSA"
}

empresas_exibicao = [

    nomes_exibicao.get(
        empresa,
        empresa
    )

    for empresa

    in taxas_mensais.index

]

######### 5.2.6 — DASHBOARD EXECUTIVO DE CANCELAMENTOS #########

# PALETA VISUAL

AZUL_PRIMARIO = "#17324D"
AZUL_SECUNDARIO = "#4F81BD"
AZUL_CLARO = "#9BC2E6"
LARANJA = "#D97745"
VERMELHO = "#C0392B"
VERMELHO_CLARO = "#E38B83"
FUNDO = "#F4F7FB"
CARD = "#FFFFFF"
TEXTO = "#24364B"
TEXTO_SECUNDARIO = "#64748B"
BORDA = "#D9E2F0"

# PADRONIZAÇÃO VISUAL DOS GRÁFICOS

def aplicar_layout(fig):

    fig.update_layout(

        template="plotly_white",

        paper_bgcolor=CARD,

        plot_bgcolor=CARD,

        font=dict(
            family="Segoe UI",
            size=12,
            color=TEXTO
        ),

        title_font=dict(
            size=22,
            color=TEXTO
        ),

        margin=dict(
            l=40,
            r=40,
            t=80,
            b=40
        ),

        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Segoe UI"
        )

    )

    fig.update_xaxes(
        showgrid=False,
        showline=True,
        linewidth=1,
        linecolor=BORDA,
        zeroline=False
    )

    fig.update_yaxes(
        showline=True,
        linewidth=1,
        linecolor=BORDA,
        zeroline=False
    )

# FIGURA 1 — VISÃO GERAL

fig_geral = go.Figure()

# BARRAS
# VOOS REALIZADOS

fig_geral.add_trace(

    go.Bar(

        x=nomes_meses_programados,

        y=resultado_cancelamento["REALIZADO"],

        name="Voos realizados",

        marker=dict(

            color=AZUL_PRIMARIO,

            line=dict(
                color="white",
                width=1
            )

        ),

        text=[
            f"{valor:,.0f}".replace(",", ".")
            for valor
            in resultado_cancelamento["REALIZADO"]
        ],

        textposition="outside",

        hovertemplate=(

            "<b>%{x}</b><br>"
            "Voos realizados: %{y:,.0f}"
            "<extra></extra>"

        )

    )

)

# LINHA / TAXA DE CANCELAMENTO


fig_geral.add_trace(

    go.Scatter(

        x=nomes_meses_programados,

        y=resultado_cancelamento[
            "TAXA_CANCELAMENTO (%)"
        ],

        mode="lines+markers+text",

        name="Taxa de cancelamento",

        text=[
            f"{valor:.2f}%"
            for valor
            in resultado_cancelamento[
                "TAXA_CANCELAMENTO (%)"
            ]
        ],

        textposition="top center",

        line=dict(

            color=VERMELHO,

            width=4

        ),

        marker=dict(

            size=12,

            color=VERMELHO,

            line=dict(
                color="white",
                width=2
            )

        ),

        yaxis="y2",

        hovertemplate=(

            "<b>%{x}</b><br>"

            "Taxa de cancelamento: "
            "<b>%{y:.2f}%</b>"

            "<extra></extra>"

        )

    )

)

# LAYOUT
fig_geral.update_layout(

    title=dict(

        text=(
            "<b>Painel Executivo de Operações</b>"
            "<br><sup>"
            "Volume operacional e taxa de cancelamento"
            "</sup>"
        ),

        x=0.5

    ),

    barmode="group",

    xaxis_title="Mês",

    yaxis=dict(

        title="Voos realizados",

        separatethousands=True

    ),

    yaxis2=dict(

    title=dict(

        text="Taxa de cancelamento (%)",

        font=dict(
            color=VERMELHO
        )

    ),

    tickfont=dict(
        color=VERMELHO
    ),

    overlaying="y",

    side="right",

    showgrid=False,

    ticksuffix="%"

    ),

    legend=dict(

        orientation="h",

        y=1.05,

        x=0.5,

        xanchor="center"

    )

)

aplicar_layout(
    fig_geral
)

fig_geral.update_yaxes(
    separatethousands=True
)

# REFERÊNCIA VISUAL

fig_geral.add_hline(

    y=3,

    yref="y2",

    line_width=2,

    line_dash="dash",

    line_color=VERMELHO,

    annotation_text="3%",

    annotation_position="top right"

)

# FIGURA 2 — VOLUME DE VOOS POR TIPO DE LINHA

fig_tipo_linha = go.Figure()

# PREPARAÇÃO DOS TOTAIS

dados_tipo_linha = resumo_tipo_linha.copy()

dados_tipo_linha["TOTAL"] = (
    dados_tipo_linha["REALIZADO"]
    + dados_tipo_linha["CANCELADO"]
)

dados_tipo_linha = dados_tipo_linha.sort_values(
    "TOTAL",
    ascending=True
)

# VOOS REALIZADOS

fig_tipo_linha.add_trace(
    go.Bar(
        y=dados_tipo_linha["Descrição Categoria"],
        x=dados_tipo_linha["REALIZADO"],

        name="Realizados",
        orientation="h",

        marker=dict(
            color=AZUL_PRIMARIO,
            line=dict(width=0)
        ),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Situação: Realizados<br>"
            "Quantidade: %{x:,.0f} voos"
            "<extra></extra>"
        )
    )
)

# VOOS CANCELADOS

fig_tipo_linha.add_trace(
    go.Bar(
        y=dados_tipo_linha["Descrição Categoria"],
        x=dados_tipo_linha["CANCELADO"],

        name="Cancelados",
        orientation="h",

        marker=dict(
            color="#C0392B",
            line=dict(width=0)
        ),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Situação: Cancelados<br>"
            "Quantidade: %{x:,.0f} voos"
            "<extra></extra>"
        )
    )
)

# TOTAL NO FINAL DE CADA BARRA

fig_tipo_linha.add_trace(
    go.Scatter(
        y=dados_tipo_linha["Descrição Categoria"],
        x=dados_tipo_linha["TOTAL"],

        mode="text",
        text=dados_tipo_linha["TOTAL"].map(
            lambda valor: f"{valor:,.0f}".replace(",", ".")
        ),

        textposition="middle right",
        textfont=dict(
            family="Arial",
            size=11,
            color=TEXTO
        ),

        name="Total",
        showlegend=False,
        cliponaxis=False,

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Total de voos: %{x:,.0f}"
            "<extra></extra>"
        )
    )
)

# LAYOUT

fig_tipo_linha.update_layout(
    title=dict(
        text=(
            "<b>Volume de Voos por Tipo de Linha</b>"
            "<br><sup>"
            "Comparação entre voos realizados e cancelados"
            "</sup>"
        ),
        x=0.5,
        xanchor="center"
    ),

    barmode="stack",

    xaxis_title="Quantidade de voos",
    yaxis_title=None,

    legend=dict(
        title="Situação",
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="center",
        x=0.5
    ),

    bargap=0.30,

    margin=dict(
        l=130,
        r=85,
        t=105,
        b=55
    ),

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#D9E2F0",
        font=dict(
            family="Arial",
            size=12,
            color="#263445"
        )
    )
)

# EIXO X
fig_tipo_linha.update_xaxes(
    title_font=dict(size=12),
    tickfont=dict(size=11),
    separatethousands=True,
    showgrid=True,
    gridcolor="#E8EEF5",
    zeroline=False,
    range=[
        0,
        dados_tipo_linha["TOTAL"].max() * 1.15
    ]
)

# EIXO Y
fig_tipo_linha.update_yaxes(
    tickfont=dict(size=11),
    categoryorder="array",
    categoryarray=dados_tipo_linha["Descrição Categoria"],
    showgrid=False,
    zeroline=False
)

# PADRÃO VISUAL DO PROJETO

aplicar_layout(fig_tipo_linha)

fig_tipo_linha.update_xaxes(
    range=[
        0,
        dados_tipo_linha["TOTAL"].max() * 1.15
    ]
)

fig_tipo_linha.update_layout(
    margin=dict(
        l=130,
        r=85,
        t=105,
        b=55
    )
)

######### FIGURA 3 — VOLUME DE OPERAÇÕES POR EMPRESA #########

# 1. PREPARAR OS DADOS

dados_empresas = (
    top_empresas
    .copy()
    .sort_values("TOTAL", ascending=False)
    .reset_index(drop=True)
)

dados_empresas["RANKING"] = np.arange(1, len(dados_empresas) + 1)
total_voos_base = len(df_tratado)
dados_empresas["PARTICIPACAO (%)"] = (
    dados_empresas["TOTAL"] / total_voos_base * 100
    if total_voos_base > 0
    else 0
)

dados_empresas_plot = dados_empresas.iloc[::-1].copy()

# ESCALA DE AZUIS CONFORME O VOLUME DE OPERAÇÕES

escala_azuis = [
    "#C6DBEF",
    "#A6C8E3",
    "#78A6D4",
    "#477FB5",
    "#174A7E"
]

quantidade_empresas = len(dados_empresas_plot)

if quantidade_empresas > 1:
    indices_cores = np.linspace(
        0,
        len(escala_azuis) - 1,
        quantidade_empresas
    ).round().astype(int)
else:
    indices_cores = [len(escala_azuis) - 1]

cores_empresas = [
    escala_azuis[indice]
    for indice in indices_cores
]

rotulos_totais = dados_empresas_plot["TOTAL"].map(
    lambda valor: f"{valor:,.0f}".replace(",", ".")
)

# 2. CRIAR O GRÁFICO

fig_empresas_dashboard = go.Figure()

fig_empresas_dashboard.add_trace(
    go.Bar(
        y=dados_empresas_plot["Empresa Aérea"],
        x=dados_empresas_plot["TOTAL"],

        orientation="h",

        marker=dict(
            color=cores_empresas,
            line=dict(width=0)
        ),

        text=rotulos_totais,
        textposition="outside",
        textfont=dict(
            family="Arial",
            size=11,
            color=TEXTO
        ),

        cliponaxis=False,

        customdata=np.column_stack([
            dados_empresas_plot["RANKING"],
            dados_empresas_plot["PARTICIPACAO (%)"]
        ]),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Posição no ranking: %{customdata[0]:.0f}º<br>"
            "Quantidade de voos: <b>%{x:,.0f}</b><br>"
            "Participação na base: <b>%{customdata[1]:.2f}%</b>"
            "<extra></extra>"
        )
    )
)

# 3. CONFIGURAR O LAYOUT

fig_empresas_dashboard.update_layout(
    title=dict(
        text=(
            "<b>Empresas com Maior Volume de Voos</b>"
            "<br><sup>"
            "15 empresas com mais operações registradas em 2025"
            "</sup>"
        ),
        x=0.02,
        xanchor="left"
    ),

    xaxis_title="Quantidade de voos",
    yaxis_title=None,

    showlegend=False,
    bargap=0.28,

    margin=dict(
        l=155,
        r=85,
        t=90,
        b=55
    ),

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#D9E2F0",
        font=dict(
            family="Arial",
            size=12,
            color="#263445"
        )
    )
)

# 4. AJUSTAR OS EIXOS

maior_volume = dados_empresas_plot["TOTAL"].max()

fig_empresas_dashboard.update_xaxes(
    separatethousands=True,
    showgrid=True,
    gridcolor="#E8EEF5",
    zeroline=False,
    tickfont=dict(size=11),
    rangemode="tozero",
    range=[0, maior_volume * 1.18]
)

fig_empresas_dashboard.update_yaxes(
    tickfont=dict(size=11),
    showgrid=False,
    zeroline=False,
    automargin=True
)

# 5. APLICAR O PADRÃO VISUAL DO PROJETO

aplicar_layout(fig_empresas_dashboard)
fig_empresas_dashboard.update_xaxes(
    range=[0, maior_volume * 1.18]
)

fig_empresas_dashboard.update_layout(
    margin=dict(
        l=155,
        r=85,
        t=90,
        b=55
    )
)

######### FIGURA 4 — TAXAS DE CANCELAMENTO #########

# 1. PREPARAR OS DADOS

valores_cancelados = dados_grafico["CANCELADO"].to_numpy(dtype=float)

max_cancelados = valores_cancelados.max() if len(valores_cancelados) > 0 else 0

if max_cancelados > 0:
    tamanhos_pontos = (
        14
        + 28 * np.sqrt(valores_cancelados / max_cancelados)
    )
else:
    tamanhos_pontos = np.full(len(valores_cancelados), 14.0)

# 2. CRIAR O GRÁFICO

fig_taxas_dashboard = go.Figure()

fig_taxas_dashboard.add_trace(
    go.Scatter(
        x=dados_grafico["TOTAL PROGRAMADO"],
        y=dados_grafico["TAXA CANCELAMENTO (%)"],

        mode="markers+text",

        text=dados_grafico["Empresa Aérea"],
        textposition="top center",

        textfont=dict(
            family="Arial",
            size=10,
            color=TEXTO
        ),

        customdata=np.column_stack([
            dados_grafico["CANCELADO"],
            dados_grafico["REALIZADO"],
            dados_grafico["TOTAL PROGRAMADO"]
        ]),

        marker=dict(
            size=tamanhos_pontos,
            sizemode="diameter",

            color=dados_grafico["TAXA CANCELAMENTO (%)"],

            colorscale=[
                [0.00, "#174A7E"],
                [0.35, "#78A6D4"],
                [0.60, "#F2C078"],
                [0.80, "#E67E45"],
                [1.00, "#C0392B"]
            ],

            cmin=0,
            cmax=max(
                30,
                dados_grafico["TAXA CANCELAMENTO (%)"].max()
            ),

            showscale=True,

            colorbar=dict(
                title=dict(
                    text="Taxa de<br>cancelamento"
                ),
                ticksuffix="%",
                thickness=13,
                len=0.72,
                outlinewidth=0,
                tickfont=dict(size=10)
            ),

            opacity=0.88,

            line=dict(
                color="white",
                width=1.5
            )
        ),

        hovertemplate=(
            "<b>%{text}</b><br><br>"
            "Taxa de cancelamento: <b>%{y:.2f}%</b><br>"
            "Voos cancelados: %{customdata[0]:,.0f}<br>"
            "Voos realizados: %{customdata[1]:,.0f}<br>"
            "Voos programados: %{customdata[2]:,.0f}"
            "<extra></extra>"
        )
    )
)

# 3. LAYOUT

fig_taxas_dashboard.update_layout(
    title=dict(
        text=(
            "<b>10 Empresas Aéreas com Maiores Taxas de Cancelamento</b>"
            "<br><sup>"
            "Empresas com pelo menos 1.000 voos programados"
            "</sup>"
        ),
        x=0.02,
        xanchor="left"
    ),

    xaxis_title="Quantidade de voos programados",
    yaxis_title="Taxa de cancelamento (%)",

    margin=dict(
        l=65,
        r=90,
        t=100,
        b=65
    ),

    hovermode="closest",

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#D9E2F0",
        font=dict(
            family="Arial",
            size=12,
            color="#263445"
        )
    )
)

# 4. AJUSTAR OS EIXOS

fig_taxas_dashboard.update_xaxes(
    separatethousands=True,
    showgrid=True,
    gridcolor="#E8EEF5",
    zeroline=False,
    rangemode="tozero",
    tickfont=dict(size=11)
)

fig_taxas_dashboard.update_yaxes(
    ticksuffix="%",
    showgrid=True,
    gridcolor="#E8EEF5",
    zeroline=False,
    rangemode="tozero",
    tickfont=dict(size=11)
)

# 5. PADRÃO VISUAL DO PROJETO

aplicar_layout(fig_taxas_dashboard)
fig_taxas_dashboard.update_layout(
    margin=dict(
        l=65,
        r=90,
        t=100,
        b=65
    )
)

# 6. LINHA DE REFERÊNCIA

fig_taxas_dashboard.add_hline(
    y=30,
    line_width=1.5,
    line_dash="dash",
    line_color=VERMELHO,

    annotation_text="Referência: 30%",
    annotation_position="top right",

    annotation_font=dict(
        family="Arial",
        size=10,
        color=VERMELHO
    )
)

######### FIGURA 5 — EVOLUÇÃO DAS TAXAS DE CANCELAMENTO #########

# 1. Ordenar empresas pela taxa média anual

ordem_empresas = (
    taxas_mensais
    .mean(axis=1)
    .sort_values(ascending=False, na_position="last")
    .index
)

taxas_mensais = taxas_mensais.loc[ordem_empresas]
cancelados_mensais = cancelados_mensais.loc[ordem_empresas]
programados_mensais = programados_mensais.loc[ordem_empresas]

# 2. Preparar os dados

empresas_exibicao = [
    nomes_exibicao.get(empresa, empresa)
    for empresa in ordem_empresas
]

valores_taxas = taxas_mensais.to_numpy(dtype=float)

valores_validos = valores_taxas[np.isfinite(valores_taxas)]

if valores_validos.size > 0:
    zmax = np.percentile(valores_validos, 95)
else:
    zmax = 1

if not np.isfinite(zmax) or zmax <= 0:
    zmax = 1

# 3. Construir o heatmap

fig_heatmap_dashboard = go.Figure()

fig_heatmap_dashboard.add_trace(
    go.Heatmap(
        z=valores_taxas,
        x=nomes_meses,
        y=empresas_exibicao,
        xgap=0,
        ygap=0,
        colorscale=[
            [0.00, "#F1F5F9"],
            [0.25, "#C7DDF0"],
            [0.50, "#78A6D4"],
            [0.75, "#E6A15A"],
            [1.00, "#C0392B"]
        ],

        zmin=0,
        zmax=zmax,
        zauto=False,

        customdata=np.stack(
            [
                cancelados_mensais.to_numpy(),
                programados_mensais.to_numpy()
            ],
            axis=-1
        ),

        colorbar=dict(
            title=dict(
                text="Taxa de<br>cancelamento (%)"
            ),
            thickness=12,
            len=0.85,
            outlinewidth=0,
            ticksuffix="%",
            tickfont=dict(size=10),
            bgcolor="rgba(255,255,255,0)"
        ),

        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mês: %{x}<br><br>"
            "Taxa de cancelamento: "
            "<b>%{z:.2f}%</b><br>"
            "Voos cancelados: %{customdata[0]:,.0f}<br>"
            "Voos programados: %{customdata[1]:,.0f}"
            "<extra></extra>"
        ),

        hoverongaps=False
    )
)

# 4. Configurar o layout

fig_heatmap_dashboard.update_layout(
    title=dict(
        text=(
            "<b>Evolução Mensal das Taxas de Cancelamento</b>"
            "<br><sup>"
            "10 Empresas com mais cancelamento ordenadas pela taxa média anual"
            "</sup>"
        ),
        x=0.02,
        xanchor="left",
        y=0.90,
        yanchor="top"
    ),

    xaxis_title=None,
    yaxis_title=None,

    margin=dict(
        l=155,
        r=75,
        t=85,
        b=45
    ),

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#D9E2F0",
        font=dict(
            family="Arial",
            size=12,
            color="#263445"
        )
    )
)
# 5. Ajustar os eixos

fig_heatmap_dashboard.update_xaxes(
    tickfont=dict(size=11),
    tickangle=0,
    showgrid=False,
    zeroline=False,
    fixedrange=False,
    side="bottom"
)

fig_heatmap_dashboard.update_yaxes(
    tickfont=dict(size=10),
    autorange="reversed",
    showgrid=False,
    zeroline=False,
    fixedrange=False,
    ticks="",
    automargin=True
)

aplicar_layout(fig_heatmap_dashboard)

######### 5.3 — VOLUMES MENSAIS DAS EMPRESAS INVESTIGADAS #########

empresas_investigadas = (
    top_10_empresas_cancelamento.index.tolist()
)

volumes_mensais = (
    resultado_mensal_empresas
    .reset_index()
    .loc[
        lambda dados: dados["Empresa Aérea"].isin(
            empresas_investigadas
        )
    ]
    .copy()
)

volumes_mensais["Mês Número"] = volumes_mensais["Mês"]

volumes_mensais["Mês"] = volumes_mensais["Mês"].map(
    dict(enumerate(nomes_meses, start=1))
)

volumes_mensais = volumes_mensais[
    [
        "Empresa Aérea",
        "Mês Número",
        "Mês",
        "REALIZADO",
        "CANCELADO",
        "TOTAL PROGRAMADO",
        "TAXA CANCELAMENTO (%)"
    ]
].sort_values(
    ["Empresa Aérea", "Mês Número"]
)

print("\nVOLUMES MENSAIS DAS EMPRESAS INVESTIGADAS")

print(
    volumes_mensais[
        [
            "Empresa Aérea",
            "Mês",
            "REALIZADO",
            "CANCELADO",
            "TOTAL PROGRAMADO",
            "TAXA CANCELAMENTO (%)"
        ]
    ].to_string(index=False)
)

meses_criticos = volumes_mensais.loc[
    volumes_mensais["TAXA CANCELAMENTO (%)"] >= 50
].sort_values(
    ["Empresa Aérea", "Mês Número"]
)

print(
    "\nMESES COM TAXA DE CANCELAMENTO "
    "IGUAL OU SUPERIOR A 50%"
)

print(
    meses_criticos[
        [
            "Empresa Aérea",
            "Mês",
            "REALIZADO",
            "CANCELADO",
            "TOTAL PROGRAMADO",
            "TAXA CANCELAMENTO (%)"
        ]
    ].to_string(index=False)
)

# ETAPA 5.3.1 — INVESTIGAÇÃO DE CANCELAMENTOS POR ROTA

companhias_investigacao = [
    "ARAJET S.A.",
    "ETHIOPIAN AIRLINES GROUP"
]

# Considerar apenas registros com programação válida,
# situação de voo válida e aeroportos identificados.
df_rotas = df_eda.loc[
    df_eda["Empresa Aérea"].isin(companhias_investigacao)
    & df_eda["Partida Prevista"].notna()
    & df_eda["Situação Voo"].isin(["REALIZADO", "CANCELADO"])
    & df_eda["Sigla ICAO Aeroporto Origem"].notna()
    & df_eda["Sigla ICAO Aeroporto Destino"].notna()
].copy()

df_rotas["Rota"] = (
    df_rotas["Sigla ICAO Aeroporto Origem"]
    + " → "
    + df_rotas["Sigla ICAO Aeroporto Destino"]
)

resumo_rotas = (
    df_rotas
    .groupby(
        ["Empresa Aérea", "Rota", "Situação Voo"]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for situacao in ["REALIZADO", "CANCELADO"]:
    if situacao not in resumo_rotas.columns:
        resumo_rotas[situacao] = 0

resumo_rotas["TOTAL PROGRAMADO"] = (
    resumo_rotas["REALIZADO"]
    + resumo_rotas["CANCELADO"]
)

resumo_rotas["TAXA CANCELAMENTO (%)"] = (
    resumo_rotas["CANCELADO"]
    .div(resumo_rotas["TOTAL PROGRAMADO"])
    .mul(100)
)

# Exigir pelo menos 30 voos programados por rota
resumo_rotas_relevantes = resumo_rotas.loc[
    resumo_rotas["TOTAL PROGRAMADO"] >= 30
].copy()

resumo_rotas_relevantes = (
    resumo_rotas_relevantes
    .sort_values(
        "TAXA CANCELAMENTO (%)",
        ascending=False
    )
    .reset_index(drop=True)
)

print("\n15 ROTAS COM MAIORES TAXAS DE CANCELAMENTO")
print("Critério: mínimo de 30 voos programados por rota\n")

print(
    resumo_rotas_relevantes[
        [
            "Empresa Aérea",
            "Rota",
            "REALIZADO",
            "CANCELADO",
            "TOTAL PROGRAMADO",
            "TAXA CANCELAMENTO (%)"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

print("\n15 ROTAS COM MAIOR QUANTIDADE DE CANCELAMENTOS\n")

print(
    resumo_rotas_relevantes
    .sort_values("CANCELADO", ascending=False)[
        [
            "Empresa Aérea",
            "Rota",
            "REALIZADO",
            "CANCELADO",
            "TOTAL PROGRAMADO",
            "TAXA CANCELAMENTO (%)"
        ]
    ]
    .head(15)
    .to_string(index=False)
)

######### FIGURA 8 — TOP 10 ROTAS COM MAIS VOOS CANCELADOS #########
# Objetivo:
# Identificar as rotas com maior impacto absoluto de cancelamentos.

# 1. PREPARAR OS DADOS

df_grafico_taxas_rota = resumo_rotas_relevantes.copy()

colunas_numericas = [
    "TOTAL PROGRAMADO",
    "CANCELADO",
    "REALIZADO",
    "TAXA CANCELAMENTO (%)",
]

for coluna in colunas_numericas:
    df_grafico_taxas_rota[coluna] = pd.to_numeric(
        df_grafico_taxas_rota[coluna],
        errors="coerce",
    )

df_grafico_taxas_rota = (
    df_grafico_taxas_rota
    .dropna(
        subset=[
            "Empresa Aérea",
            "Rota",
            "TOTAL PROGRAMADO",
            "CANCELADO",
            "REALIZADO",
            "TAXA CANCELAMENTO (%)",
        ]
    )
    .copy()
)

df_grafico_taxas_rota = df_grafico_taxas_rota.loc[
    (df_grafico_taxas_rota["TOTAL PROGRAMADO"] > 0)
    & (df_grafico_taxas_rota["CANCELADO"] > 0)
    & (df_grafico_taxas_rota["REALIZADO"] >= 0)
    & df_grafico_taxas_rota["TAXA CANCELAMENTO (%)"].between(0, 100)
].copy()

if df_grafico_taxas_rota.empty:
    raise ValueError(
        "Não existem rotas válidas com voos cancelados "
        "para construir a Figura 8."
    )

# Selecionar as 10 combinações com maior quantidade de cancelamentos.
df_top10_rotas_cancelamento = (
    df_grafico_taxas_rota
    .sort_values(
        by=[
            "CANCELADO",
            "TAXA CANCELAMENTO (%)",
            "TOTAL PROGRAMADO",
        ],
        ascending=[False, False, False],
    )
    .head(10)
    .copy()
)
df_top10_rotas_cancelamento["ROTA_EMPRESA"] = (
    df_top10_rotas_cancelamento["Empresa Aérea"].astype(str)
    + " | "
    + df_top10_rotas_cancelamento["Rota"].astype(str)
)

# Ordenação para exibir a maior quantidade no topo do gráfico horizontal.
df_top10_rotas_cancelamento = (
    df_top10_rotas_cancelamento
    .sort_values("CANCELADO", ascending=True)
    .copy()
)

# 2. CRIAR O GRÁFICO

fig_taxa_cancelamento_rota = go.Figure()

COR_AZUL_CLARO = "#9BC2E6"
COR_AZUL_MEDIO = "#4F8FC5"
COR_AZUL_ESCURO = "#174A7E"
COR_VERMELHO = "#D9534F"
COR_TEXTO = "#263445"
COR_GRADE = "#E8EEF5"

# Separar as rotas com 100% de cancelamento.
mask_cancelamento_total = (
    df_top10_rotas_cancelamento["TAXA CANCELAMENTO (%)"].eq(100)
)

df_rotas_demais = df_top10_rotas_cancelamento.loc[
    ~mask_cancelamento_total
].copy()

df_rotas_cancelamento_total = df_top10_rotas_cancelamento.loc[
    mask_cancelamento_total
].copy()

# 2.1. ROTAS COM TAXA INFERIOR A 100%

if not df_rotas_demais.empty:
    fig_taxa_cancelamento_rota.add_trace(
        go.Bar(
            x=df_rotas_demais["CANCELADO"],
            y=df_rotas_demais["ROTA_EMPRESA"],
            orientation="h",
            name="Taxa inferior a 100%",
            marker=dict(
                color=df_rotas_demais["TAXA CANCELAMENTO (%)"],
                colorscale=[
                    [0.00, COR_AZUL_CLARO],
                    [0.50, COR_AZUL_MEDIO],
                    [1.00, COR_AZUL_ESCURO],
                ],
                cmin=0,
                cmax=100,
                line=dict(
                    color="rgba(23,74,126,0.35)",
                    width=0.8,
                ),
                cornerradius=4,
            ),
            customdata=df_rotas_demais[
                [
                    "TOTAL PROGRAMADO",
                    "REALIZADO",
                    "TAXA CANCELAMENTO (%)",
                ]
            ].to_numpy(),
            text=df_rotas_demais["CANCELADO"].map(
                lambda valor: f"{valor:,.0f}".replace(",", ".")
            ),
            textposition="outside",
            textfont=dict(
                family="Arial",
                size=11,
                color=COR_TEXTO,
            ),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Voos cancelados: %{x:,.0f}"
                "<br>Voos programados: %{customdata[0]:,.0f}"
                "<br>Voos realizados: %{customdata[1]:,.0f}"
                "<br>Taxa de cancelamento: %{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

# 2.2. ROTAS COM 100% DE CANCELAMENTO

if not df_rotas_cancelamento_total.empty:
    fig_taxa_cancelamento_rota.add_trace(
        go.Bar(
            x=df_rotas_cancelamento_total["CANCELADO"],
            y=df_rotas_cancelamento_total["ROTA_EMPRESA"],
            orientation="h",
            name="100% de cancelamento",
            marker=dict(
                color=COR_VERMELHO,
                line=dict(
                    color="#B94040",
                    width=1,
                ),
                cornerradius=4,
            ),
            customdata=df_rotas_cancelamento_total[
                [
                    "TOTAL PROGRAMADO",
                    "REALIZADO",
                    "TAXA CANCELAMENTO (%)",
                ]
            ].to_numpy(),
            text=df_rotas_cancelamento_total["CANCELADO"].map(
                lambda valor: f"{valor:,.0f}".replace(",", ".")
            ),
            textposition="outside",
            textfont=dict(
                family="Arial",
                size=11,
                color=COR_TEXTO,
            ),
            cliponaxis=False,
            hovertemplate=(
                "<b>%{y}</b>"
                "<br>Voos cancelados: %{x:,.0f}"
                "<br>Voos programados: %{customdata[0]:,.0f}"
                "<br>Voos realizados: %{customdata[1]:,.0f}"
                "<br>Taxa de cancelamento: %{customdata[2]:.2f}%"
                "<extra></extra>"
            ),
        )
    )

# 3. CONFIGURAR O LAYOUT

COR_PRINCIPAL = "#174A7E"
COR_TEXTO = "#263445"
COR_GRADE = "#E8EEF5"

aplicar_layout(fig_taxa_cancelamento_rota)

fig_taxa_cancelamento_rota.update_layout(
    title=dict(
        text=(
            "<b>Top 10 Rotas com Mais Voos Cancelados</b>"
            "<br>"
            "<span style='font-size:13px; font-weight:normal;'>"
            "Empresas e rotas com pelo menos 30 voos programados em 2025"
            "</span>"
        ),
        x=0.02,
        xanchor="left",
        y=0.91,
        yanchor="top",
        font=dict(
            family="Arial",
            size=21,
            color="#263445",
        ),
    ),

    template="plotly_white",
    height=620,
    showlegend=True,
    hovermode="closest",
    bargap=0.32,

    font=dict(
        family="Arial",
        size=12,
        color="#263445",
    ),

    xaxis=dict(
        title=dict(
            text="Quantidade de voos cancelados",
            font=dict(size=13, color=COR_TEXTO),
        ),
        showgrid=True,
        gridcolor="#263445",
        gridwidth=1,
        zeroline=False,
        rangemode="tozero",
        ticks="outside",
        tickfont=dict(size=11, color=COR_TEXTO),
        showline=False,
    ),

    yaxis=dict(
        title=None,
        showgrid=False,
        zeroline=False,
        ticks="",
        tickfont=dict(
            family="Arial",
            size=11,
            color="#263445",
        ),
        automargin=True,
    ),

    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor=COR_PRINCIPAL,
        font_size=12,
        font_family="Arial",
        font_color="#263445",
        align="left",
    ),

    margin=dict(
        l=220,
        r=75,
        t=145,
        b=75,
    ),

    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",

    uniformtext=dict(
        minsize=10,
        mode="show",
    ),
)

# 4. RESUMO ANALÍTICO NO CONSOLE

total_rotas_analisadas = len(df_grafico_taxas_rota)

total_cancelamentos_top10 = int(
    df_top10_rotas_cancelamento["CANCELADO"].sum()
)

maior_numero_cancelamentos = int(
    df_top10_rotas_cancelamento["CANCELADO"].max()
)

rota_maior_impacto = (
    df_top10_rotas_cancelamento
    .sort_values("CANCELADO", ascending=False)
    .iloc[0]
)

print("\n" + "=" * 70)
print("FIGURA 8 — TOP 10 ROTAS COM MAIOR NÚMERO DE CANCELAMENTOS")
print("=" * 70)

print(
    f"\nCombinações de empresa e rota elegíveis: "
    f"{total_rotas_analisadas:,}"
)

print(
    f"Rotas exibidas no gráfico: "
    f"{len(df_top10_rotas_cancelamento):,}"
)

print(
    f"Total de cancelamentos nas 10 rotas: "
    f"{total_cancelamentos_top10:,}"
)

print(
    "\nCombinação com maior número de cancelamentos:"
)

print(
    f"Empresa: {rota_maior_impacto['Empresa Aérea']}"
)

print(
    f"Rota: {rota_maior_impacto['Rota']}"
)

print(
    f"Voos cancelados: {maior_numero_cancelamentos:,}"
)

print(
    f"Voos programados: "
    f"{int(rota_maior_impacto['TOTAL PROGRAMADO']):,}"
)

print(
    f"Taxa de cancelamento: "
    f"{rota_maior_impacto['TAXA CANCELAMENTO (%)']:.2f}%"
)

print(
    "\nInterpretação:"
    "\n- Cada barra representa uma combinação de empresa aérea e rota."
    "\n- As barras estão ordenadas pela quantidade de cancelamentos."
    "\n- A cor representa a taxa de cancelamento."
    "\n- Passe o mouse sobre uma barra para consultar o volume "
    "programado, os voos realizados e a taxa."
)

print("\nGráfico criado: fig_taxa_cancelamento_rota")

######### ETAPA 5.3.2 — VALIDAÇÃO DAS ROTAS COM CANCELAMENTO ELEVADO #########

rotas_investigacao = [
    ("ARAJET S.A.", "MDPC", "SAEZ"),
    ("ARAJET S.A.", "SAEZ", "MDPC"),
    ("ETHIOPIAN AIRLINES GROUP", "DNMM", "SBGR"),
    ("ETHIOPIAN AIRLINES GROUP", "SBGR", "SCEL"),
]

mask_rotas_investigadas = pd.Series(
    False,
    index=df_eda.index,
)

for empresa, origem, destino in rotas_investigacao:
    mask_rotas_investigadas |= (
        df_eda["Empresa Aérea"].eq(empresa)
        & df_eda["Sigla ICAO Aeroporto Origem"].eq(origem)
        & df_eda["Sigla ICAO Aeroporto Destino"].eq(destino)
    )

df_rotas_suspeitas = df_eda.loc[
    mask_rotas_investigadas
    & df_eda["Partida Prevista"].notna()
    & df_eda["Situação Voo"].isin(["REALIZADO", "CANCELADO"])
].copy()

# Extrair o mês a partir da partida prevista.
df_rotas_suspeitas["Mês"] = (
    df_rotas_suspeitas["Partida Prevista"].dt.month
)

# Resumir os voos por empresa, rota, mês e situação.
resumo_mensal_rotas = (
    df_rotas_suspeitas
    .groupby(
        [
            "Empresa Aérea",
            "Sigla ICAO Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Mês",
            "Situação Voo",
        ]
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for situacao in ["REALIZADO", "CANCELADO"]:
    if situacao not in resumo_mensal_rotas.columns:
        resumo_mensal_rotas[situacao] = 0

# Calcular total programado e taxa mensal de cancelamento.
resumo_mensal_rotas["TOTAL PROGRAMADO"] = (
    resumo_mensal_rotas["REALIZADO"]
    + resumo_mensal_rotas["CANCELADO"]
)

resumo_mensal_rotas["TAXA CANCELAMENTO (%)"] = (
    resumo_mensal_rotas["CANCELADO"]
    .div(
        resumo_mensal_rotas["TOTAL PROGRAMADO"].replace(
            0, np.nan
        )
    )
    .mul(100)
)

print("\nCANCELAMENTOS POR MÊS NAS QUATRO ROTAS INVESTIGADAS\n")

print(
    resumo_mensal_rotas
    .sort_values(
        [
            "Empresa Aérea",
            "Sigla ICAO Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Mês",
        ]
    )
    .to_string(index=False)
)

######### 5.3.3 — DETALHAMENTO DOS REGISTROS POR VOO E ROTA #########

print("\nDETALHAMENTO DOS REGISTROS POR VOO E ROTA\n")

colunas_detalhamento = [
    "Empresa Aérea",
    "Número Voo",
    "Código Tipo Linha",
    "Sigla ICAO Aeroporto Origem",
    "Descrição Aeroporto Origem",
    "Sigla ICAO Aeroporto Destino",
    "Descrição Aeroporto Destino",
    "Partida Prevista",
    "Situação Voo"
]

detalhamento_rotas = (
    df_rotas_suspeitas[colunas_detalhamento]
    .groupby(
        [
            "Empresa Aérea",
            "Número Voo",
            "Código Tipo Linha",
            "Sigla ICAO Aeroporto Origem",
            "Descrição Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Descrição Aeroporto Destino",
            "Situação Voo"
        ],
        dropna=False
    )
    .agg(
        QUANTIDADE=("Partida Prevista", "size"),
        PRIMEIRA_PARTIDA=("Partida Prevista", "min"),
        ÚLTIMA_PARTIDA=("Partida Prevista", "max")
    )
    .reset_index()
    .sort_values(
        ["Empresa Aérea", "QUANTIDADE"],
        ascending=[True, False]
    )
)

print(detalhamento_rotas.to_string(index=False))

######### ETAPA 5.3.4 — CANCELAMENTOS POR TIPO DE LINHA #########

df_tipos_linha = df_rotas_suspeitas.copy()

resumo_tipos_linha = (
    df_tipos_linha
    .groupby(
        [
            "Empresa Aérea",
            "Sigla ICAO Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Código Tipo Linha",
            "Situação Voo"
        ],
        dropna=False
    )
    .size()
    .unstack(fill_value=0)
    .reset_index()
)

for situacao in ["REALIZADO", "CANCELADO"]:
    if situacao not in resumo_tipos_linha.columns:
        resumo_tipos_linha[situacao] = 0

resumo_tipos_linha["TOTAL PROGRAMADO"] = (
    resumo_tipos_linha["REALIZADO"]
    + resumo_tipos_linha["CANCELADO"]
)

resumo_tipos_linha["TAXA CANCELAMENTO (%)"] = (
    resumo_tipos_linha["CANCELADO"]
    .div(resumo_tipos_linha["TOTAL PROGRAMADO"])
    .mul(100)
)

print("\nCANCELAMENTOS POR TIPO DE LINHA E ROTA\n")

print(
    resumo_tipos_linha
    .sort_values(
        ["Empresa Aérea", "TOTAL PROGRAMADO"],
        ascending=[True, False]
    )
    .to_string(index=False)
)

print("\nDISTRIBUIÇÃO DOS TIPOS DE LINHA\n")

print(
    df_tipos_linha["Código Tipo Linha"]
    .value_counts(dropna=False)
    .to_string()
)

######### ETAPA 5.3.5 — INVESTIGAÇÃO DA CONSISTÊNCIA DOS REGISTROS DAS ROTAS #########

rotas_investigadas_consistencia = [
    ("ARAJET S.A.", "MDPC", "SAEZ"),
    ("ARAJET S.A.", "SAEZ", "MDPC"),
    ("ETHIOPIAN AIRLINES GROUP", "DNMM", "SBGR"),
    ("ETHIOPIAN AIRLINES GROUP", "SBGR", "SCEL"),
]

mask_consistencia = pd.Series(
    False,
    index=df_eda.index
)

for empresa, origem, destino in rotas_investigadas_consistencia:
    mask_consistencia |= (
        df_eda["Empresa Aérea"].eq(empresa)
        & df_eda["Sigla ICAO Aeroporto Origem"].eq(origem)
        & df_eda["Sigla ICAO Aeroporto Destino"].eq(destino)
    )

df_investigacao = df_eda.loc[
    mask_consistencia
].copy()

df_investigacao["DATA PREVISTA"] = (
    df_investigacao["Partida Prevista"]
)

df_investigacao["DIA DA SEMANA"] = (
    df_investigacao["DATA PREVISTA"].dt.day_name()
)

df_investigacao["DIA DO MÊS"] = (
    df_investigacao["DATA PREVISTA"].dt.day
)

df_investigacao["HORA PREVISTA"] = (
    df_investigacao["DATA PREVISTA"].dt.hour
)

# Resumo por empresa, rota e situação.
resumo_consistencia = (
    df_investigacao
    .groupby(
        [
            "Empresa Aérea",
            "Sigla ICAO Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Situação Voo"
        ],
        dropna=False
    )
    .agg(
        REGISTROS=("Situação Voo", "size"),
        DATAS_PREVISTAS=("DATA PREVISTA", "count"),
        DIAS_DISTINTOS=(
            "DATA PREVISTA",
            lambda x: x.dt.date.nunique()
        ),
        HORARIOS_DISTINTOS=("HORA PREVISTA", "nunique")
    )
    .reset_index()
)

print("\nRESUMO DE CONSISTÊNCIA DOS REGISTROS\n")
print(resumo_consistencia.to_string(index=False))

######### 5.3.5.1 — REGISTROS SEM PARTIDA PREVISTA #########

print("\nREGISTROS SEM PARTIDA PREVISTA\n")

registros_sem_partida_prevista = (
    df_investigacao
    .groupby(
        [
            "Empresa Aérea",
            "Sigla ICAO Aeroporto Origem",
            "Sigla ICAO Aeroporto Destino",
            "Situação Voo"
        ],
        dropna=False
    )["Partida Prevista"]
    .agg(
        REGISTROS="size",
        SEM_PARTIDA_PREVISTA=lambda x: x.isna().sum()
    )
    .reset_index()
)

registros_sem_partida_prevista = (
    registros_sem_partida_prevista.loc[
        registros_sem_partida_prevista["SEM_PARTIDA_PREVISTA"] > 0
    ]
)

print(
    registros_sem_partida_prevista.to_string(index=False)
)

######### ETAPA 5.4 — CLASSIFICAÇÕES DE PONTUALIDADE DA ANAC #########

df_pontualidade = df_voos_realizados.copy()

df_pontualidade["Classificação Partida"] = (
    padronizar_categoria(
        df_pontualidade["Situação Partida"]
    )
    .fillna("Sem classificação")
)

df_pontualidade["Classificação Chegada"] = (
    padronizar_categoria(
        df_pontualidade["Situação Chegada"]
    )
    .fillna("Sem classificação")
)

def resumir_classificacao(serie):

    quantidade = serie.value_counts(dropna=False)

    percentual = (
        quantidade
        .div(len(serie) if len(serie) else np.nan)
        .mul(100)
        .round(2)
    )

    return (
        pd.DataFrame(
            {
                "Quantidade": quantidade,
                "Percentual (%)": percentual
            }
        )
        .sort_values(
            "Quantidade",
            ascending=False
        )
    )

resultado_partida = resumir_classificacao(
    df_pontualidade["Classificação Partida"]
)

resultado_chegada = resumir_classificacao(
    df_pontualidade["Classificação Chegada"]
)

print("\nCLASSIFICAÇÃO DAS PARTIDAS — VOOS REALIZADOS")
print(resultado_partida.to_string())

print("\nCLASSIFICAÇÃO DAS CHEGADAS — VOOS REALIZADOS")
print(resultado_chegada.to_string())

print(
    "\nTotal de voos realizados analisados:",
    f"{len(df_pontualidade):,}"
)

print(
    "Nota: os percentuais usam todos os voos realizados "
    "como denominador, incluindo os sem classificação."
)

######### 5.4.1 — COMPARAÇÃO DAS CLASSIFICAÇÕES #########

categorias_principais = [
    "Antecipado",
    "Pontual",
    "Atraso 30-60",
    "Atraso 60-120",
    "Atraso 120-240",
    "Atraso > 240",
    "Sem classificação"
]
serie_partida = df_pontualidade["Classificação Partida"]
serie_chegada = df_pontualidade["Classificação Chegada"]

categorias_observadas = (
    set(serie_partida.dropna().unique())
    | set(serie_chegada.dropna().unique())
)
categorias_adicionais = sorted(
    categoria
    for categoria in categorias_observadas
    if categoria not in categorias_principais
)

categorias_grafico = categorias_principais + categorias_adicionais

partidas_percentual = (
    serie_partida.value_counts(normalize=True)
    .mul(100)
    .reindex(categorias_grafico, fill_value=0)
)

chegadas_percentual = (
    serie_chegada.value_counts(normalize=True)
    .mul(100)
    .reindex(categorias_grafico, fill_value=0)
)

fig_classificacoes = go.Figure()

fig_classificacoes.add_trace(
    go.Bar(
        y=categorias_grafico,
        x=partidas_percentual.values,
        name="Partidas",
        orientation="h",
        marker=dict(color="#174A7E", line=dict(width=0)),
        text=[
            f"{valor:.1f}%".replace(".", ",") if valor >= 1 else ""
            for valor in partidas_percentual.values
        ],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Etapa: Partidas<br>"
            "Percentual: %{x:.2f}%<extra></extra>"
        )
    )
)

fig_classificacoes.add_trace(
    go.Bar(
        y=categorias_grafico,
        x=chegadas_percentual.values,
        name="Chegadas",
        orientation="h",
        marker=dict(color="#E6A15A", line=dict(width=0)),
        text=[
            f"{valor:.1f}%".replace(".", ",") if valor >= 1 else ""
            for valor in chegadas_percentual.values
        ],
        textposition="outside",
        cliponaxis=False,
        hovertemplate=(
            "<b>%{y}</b><br>Etapa: Chegadas<br>"
            "Percentual: %{x:.2f}%<extra></extra>"
        )
    )
)

fig_classificacoes.update_layout(
    title=dict(
        text=(
            "<b>Classificações de Pontualidade: Partidas x Chegadas</b>"
            "<br><sup>Distribuição percentual dos voos realizados em 2025</sup>"
        ),
        x=0.02,
        xanchor="left"
    ),
    xaxis_title="Percentual dos voos (%)",
    yaxis_title=None,
    barmode="group",
    template="plotly_white",
    height=max(480, len(categorias_grafico) * 65),
    font=dict(family="Arial", size=12, color="#263445"),
    legend=dict(
        title="Etapa do voo",
        orientation="h",
        y=1.02,
        x=1,
        xanchor="right",
        yanchor="bottom"
    ),
    margin=dict(l=145, r=75, t=105, b=65),
    bargap=0.25,
    bargroupgap=0.08,
    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Arial"
    )
)

fig_classificacoes.update_xaxes(
    rangemode="tozero",
    ticksuffix="%",
    gridcolor="#E8EDF2",
    zeroline=False
)

fig_classificacoes.update_yaxes(
    autorange="reversed",
    showgrid=False,
    categoryorder="array",
    categoryarray=categorias_grafico
)

######### 5.4.2 — PONTUALIDADE SOMENTE ENTRE CLASSIFICADOS #########

for nome, coluna in [
    ("PARTIDA", "Classificação Partida"),
    ("CHEGADA", "Classificação Chegada")
]:

    serie = df_pontualidade[coluna]

    avaliaveis = (
        serie[
            serie.ne("Sem classificação")
        ]
    )

    print(
        f"\nCLASSIFICAÇÃO DE {nome} — SOMENTE PREENCHIDOS"
    )

    print(
        f"Registros com classificação: {len(avaliaveis):,}"
    )

    if len(avaliaveis) > 0:

        resumo = (
            avaliaveis
            .value_counts(normalize=True)
            .mul(100)
            .round(2)
        )

        print(resumo.to_string())

    else:
        print(
            "Não há classificações preenchidas para avaliar."
        )

######### 5.4.3 — AGRUPAMENTO DAS CLASSIFICAÇÕES #########

def agrupar_pontualidade(categoria):
    """
    Agrupa as classificações ANAC em:
    Antecipado, Pontual e Atrasado.
    """

    if pd.isna(categoria):
        return "Sem classificação"

    categoria = str(categoria).strip()

    if categoria == "Antecipado":
        return "Antecipado"

    if categoria == "Pontual":
        return "Pontual"

    if categoria.startswith("Atraso"):
        return "Atrasado"

    return "Sem classificação"


print(
    "\nCLASSIFICAÇÕES DISPONÍVEIS PARA O RANKING"
)

print(
    df_pontualidade[
        "Classificação Partida"
    ].value_counts(dropna=False)
)

print(
    df_pontualidade[
        "Classificação Partida"
    ].value_counts(dropna=False)
)

######### ETAPA 5.4.4 — TRANSIÇÃO DA PONTUALIDADE: PARTIDA × CHEGADA #########

# 1. DEFINIR A ORDEM E AS CORES DAS CLASSIFICAÇÕES

ordem_pontualidade = [
    "Antecipado",
    "Pontual",
    "Atrasado",
    "Sem classificação",
]

cores_pontualidade = {
    "Antecipado": "#2E8B57",
    "Pontual": "#174A7E",
    "Atrasado": "#C0392B",
    "Sem classificação": "#94A3B8",
}

# 2. PREPARAR AS CLASSIFICAÇÕES DE PARTIDA E CHEGADA

df_sankey = df_pontualidade.copy()

df_sankey["Grupo Partida"] = (
    df_sankey["Classificação Partida"]
    .apply(agrupar_pontualidade)
)

df_sankey["Grupo Chegada"] = (
    df_sankey["Classificação Chegada"]
    .apply(agrupar_pontualidade)
)

df_sankey["Grupo Partida"] = (
    df_sankey["Grupo Partida"]
    .where(
        df_sankey["Grupo Partida"].isin(ordem_pontualidade),
        "Sem classificação",
    )
    .fillna("Sem classificação")
)

df_sankey["Grupo Chegada"] = (
    df_sankey["Grupo Chegada"]
    .where(
        df_sankey["Grupo Chegada"].isin(ordem_pontualidade),
        "Sem classificação",
    )
    .fillna("Sem classificação")
)

# 3. CALCULAR OS FLUXOS ENTRE AS CLASSIFICAÇÕES

fluxos_sankey = (
    df_sankey
    .groupby(
        ["Grupo Partida", "Grupo Chegada"],
        dropna=False,
    )
    .size()
    .rename("Quantidade")
    .reset_index()
)

mapa_ordem = {
    categoria: indice
    for indice, categoria in enumerate(ordem_pontualidade)
}

fluxos_sankey["Ordem Partida"] = (
    fluxos_sankey["Grupo Partida"].map(mapa_ordem)
)

fluxos_sankey["Ordem Chegada"] = (
    fluxos_sankey["Grupo Chegada"].map(mapa_ordem)
)

fluxos_sankey = (
    fluxos_sankey
    .sort_values(
        ["Ordem Partida", "Ordem Chegada"]
    )
    .reset_index(drop=True)
)

total_fluxos_sankey = int(fluxos_sankey["Quantidade"].sum())

if total_fluxos_sankey > 0:
    fluxos_sankey["Percentual"] = (
        fluxos_sankey["Quantidade"]
        .div(total_fluxos_sankey)
        .mul(100)
    )
else:
    fluxos_sankey["Percentual"] = 0.0

# 4. PREPARAR OS NÓS E AS LIGAÇÕES DO SANKEY

mapa_origem = {
    categoria: indice
    for indice, categoria in enumerate(ordem_pontualidade)
}

mapa_destino = {
    categoria: indice + len(ordem_pontualidade)
    for indice, categoria in enumerate(ordem_pontualidade)
}

rotulos_sankey = (
    ordem_pontualidade
    + ordem_pontualidade
)

cores_nos_sankey = [
    cores_pontualidade[categoria]
    for categoria in rotulos_sankey
]

######### 5. DEFINIR A TRANSPARÊNCIA DAS LIGAÇÕES #########

def cor_fluxo_sankey(categoria):
    """
    Retorna a cor da classificação de partida com transparência,
    permitindo distinguir os fluxos sem comprometer a leitura.
    """
    cor = cores_pontualidade[categoria].lstrip("#")

    vermelho, verde, azul = (
        int(cor[i:i + 2], 16)
        for i in (0, 2, 4)
    )

    return f"rgba({vermelho},{verde},{azul},0.48)"


cores_fluxos_sankey = [
    cor_fluxo_sankey(categoria)
    for categoria in fluxos_sankey["Grupo Partida"]
]

origens_sankey = [
    mapa_origem[categoria]
    for categoria in fluxos_sankey["Grupo Partida"]
]

destinos_sankey = [
    mapa_destino[categoria]
    for categoria in fluxos_sankey["Grupo Chegada"]
]

quantidades_sankey = (
    fluxos_sankey["Quantidade"].tolist()
)

detalhes_fluxos_sankey = [
    [
        linha["Grupo Partida"],
        linha["Grupo Chegada"],
        float(linha["Percentual"]),
    ]
    for _, linha in fluxos_sankey.iterrows()
]

# 6. CONSTRUIR O GRÁFICO DE FLUXO OPERACIONAL

fig_sankey_pontualidade = go.Figure(
    data=[
        go.Sankey(
            arrangement="fixed",
            domain=dict(
                x=[0.06, 0.98],
                y=[0.04, 0.80],
            ),

            node=dict(
                pad=24,
                thickness=24,

                line=dict(
                    color="white",
                    width=1.5,
                ),

                label=rotulos_sankey,
                color=cores_nos_sankey,

                x=[0.04] * 4 + [0.91] * 4,
                y=[
                    0.05,
                    0.29,
                    0.53,
                    0.60,
                    0.05,
                    0.29,
                    0.53,
                    0.60,
                ],

                hovertemplate=(
                    "<b>%{label}</b>"
                    "<extra></extra>"
                ),
            ),

            link=dict(
                source=origens_sankey,
                target=destinos_sankey,
                value=quantidades_sankey,
                color=cores_fluxos_sankey,
                customdata=detalhes_fluxos_sankey,

                hovertemplate=(
                    "<b>%{customdata[0]} → %{customdata[1]}</b>"
                    "<br>Quantidade de voos: "
                    "<b>%{value:,.0f}</b>"
                    "<br>Participação no total: "
                    "%{customdata[2]:.2f}%"
                    "<extra></extra>"
                ),
            ),
        )
    ]
)

# 7. REFINAR O TÍTULO, AS COLUNAS E O LAYOUT

fig_sankey_pontualidade.update_layout(

    title=dict(
        text=(
            "<b>Fluxo operacional da pontualidade</b>"
            "<br>"
            "<sup>Evolução da classificação dos voos entre a partida e a chegada</sup>"
        ),
        x=0.02,
        xanchor="left",
        y=0.90,
        yanchor="top",

        font=dict(
            size=20,
            color="#1B263B",
        ),
    ),

    annotations=[

        # Rótulo acima da coluna esquerda.
        dict(
            text="<b>PARTIDA</b>",
            x=0.01,
            y=0.86,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="middle",
            font=dict(
                size=12,
                color="#174A7E",
            ),
        ),

        # Rótulo acima da coluna direita.
        dict(
            text="<b>CHEGADA</b>",
            x=0.98,
            y=0.86,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
            yanchor="middle",
            font=dict(
                size=12,
                color="#174A7E",
            ),
        ),

        # Nota explicativa mais próxima do diagrama.
        dict(
            text=(
                "A espessura das ligações representa a quantidade de voos. "
                "Passe o cursor sobre as conexões para consultar os detalhes."
            ),
            x=0.03,
            y=-0.01,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            yanchor="top",
            font=dict(
                size=10,
                color="#64748B",
            ),
        ),

    ],

    template="plotly_white",

    # Reduzir a altura total para eliminar espaço desnecessário.
    height=440,

    margin=dict(
        l=25,
        r=25,
        t=145,
        b=0,
    ),

    paper_bgcolor=CARD,
    plot_bgcolor=CARD,

    font=dict(
        family="Segoe UI",
        size=12,
        color=TEXTO,
    ),

    hoverlabel=dict(
        bgcolor="white",
        bordercolor="#CBD5E1",
        font=dict(
            family="Segoe UI",
            size=12,
            color="#24364B",
        ),
    ),
)

aplicar_layout(fig_sankey_pontualidade)


######### ETAPA 5.4.5 — MAPA DAS ROTAS COM MAIOR PERCENTUAL DE ATRASOS #########
# Objetivo:
# Identificar as rotas com maior percentual de partidas atrasadas.

from plotly.colors import sample_colorscale

# 1. VALIDAR E PREPARAR OS DADOS

df_mapa_atrasos = df_pontualidade.copy()

col_origem = "Sigla ICAO Aeroporto Origem"
col_destino = "Sigla ICAO Aeroporto Destino"

colunas_obrigatorias = [
    col_origem,
    col_destino,
    "Classificação Partida",
    "Situação Voo",
]

colunas_ausentes = [
    coluna
    for coluna in colunas_obrigatorias
    if coluna not in df_mapa_atrasos.columns
]

if colunas_ausentes:
    raise ValueError(
        f"Colunas necessárias não encontradas: {colunas_ausentes}"
    )

# Considerar somente voos realizados.
df_mapa_atrasos = df_mapa_atrasos.loc[
    df_mapa_atrasos["Situação Voo"]
    .astype("string")
    .str.strip()
    .str.upper()
    .eq("REALIZADO")
].copy()

# Padronizar os códigos ICAO.
for coluna in [col_origem, col_destino]:
    df_mapa_atrasos[coluna] = (
        df_mapa_atrasos[coluna]
        .astype("string")
        .str.strip()
        .str.upper()
    )

# Excluir rotas com códigos ausentes ou inválidos.
df_mapa_atrasos = df_mapa_atrasos.loc[
    df_mapa_atrasos[col_origem].notna()
    & df_mapa_atrasos[col_destino].notna()
    & df_mapa_atrasos[col_origem].ne("")
    & df_mapa_atrasos[col_destino].ne("")
    & df_mapa_atrasos[col_origem].ne(df_mapa_atrasos[col_destino])
].copy()

# Agrupar as classificações existentes.
df_mapa_atrasos["Grupo Partida"] = (
    df_mapa_atrasos["Classificação Partida"]
    .apply(agrupar_pontualidade)
)

df_mapa_atrasos["Grupo Partida"] = (
    df_mapa_atrasos["Grupo Partida"]
    .where(
        df_mapa_atrasos["Grupo Partida"].isin(
            ["Antecipado", "Pontual", "Atrasado"]
        ),
        "Sem classificação",
    )
)

# 2. CALCULAR OS INDICADORES POR ROTA

resumo_rotas_mapa = (
    df_mapa_atrasos
    .groupby([col_origem, col_destino], as_index=False)
    .agg(
        Total_Voos=("Grupo Partida", "size"),
        Voos_Atrasados=(
            "Grupo Partida",
            lambda s: s.eq("Atrasado").sum(),
        ),
        Voos_Classificados=(
            "Grupo Partida",
            lambda s: s.isin(
                ["Antecipado", "Pontual", "Atrasado"]
            ).sum(),
        ),
    )
)

# O denominador contém somente partidas classificadas.
resumo_rotas_mapa["Percentual_Atrasado"] = np.where(
    resumo_rotas_mapa["Voos_Classificados"] > 0,
    (
        resumo_rotas_mapa["Voos_Atrasados"]
        / resumo_rotas_mapa["Voos_Classificados"]
        * 100
    ),
    np.nan,
)

# Mínimo de 30 observações válidas para o indicador.
MINIMO_VOOS_ROTA = 30
TOP_ROTAS = 25

resumo_rotas_mapa = resumo_rotas_mapa.loc[
    (resumo_rotas_mapa["Voos_Classificados"] >= MINIMO_VOOS_ROTA)
    & resumo_rotas_mapa["Percentual_Atrasado"].notna()
].copy()

if resumo_rotas_mapa.empty:
    raise ValueError(
        "Nenhuma rota possui pelo menos "
        f"{MINIMO_VOOS_ROTA} partidas classificadas."
    )

# 3. CARREGAR A BASE GEOGRÁFICA

url_aeroportos = (
    "https://davidmegginson.github.io/"
    "ourairports-data/airports.csv"
)

url_paises = (
    "https://davidmegginson.github.io/"
    "ourairports-data/countries.csv"
)

aeroportos = pd.read_csv(
    url_aeroportos,
    usecols=[
        "icao_code",
        "type",
        "name",
        "municipality",
        "latitude_deg",
        "longitude_deg",
        "iso_country",
    ],
    low_memory=False,
)

paises = pd.read_csv(
    url_paises,
    usecols=["code", "name"],
)

# Manter aeroportos pequenos, médios e grandes.
aeroportos = aeroportos.loc[
    aeroportos["type"].isin(
        ["large_airport", "medium_airport", "small_airport"]
    )
].copy()

aeroportos["Codigo ICAO"] = (
    aeroportos["icao_code"]
    .astype("string")
    .str.strip()
    .str.upper()
)

aeroportos = aeroportos.loc[
    aeroportos["Codigo ICAO"].notna()
    & aeroportos["Codigo ICAO"].ne("")
].copy()

# Converter e validar as coordenadas.
aeroportos["Latitude"] = pd.to_numeric(
    aeroportos["latitude_deg"],
    errors="coerce",
)

aeroportos["Longitude"] = pd.to_numeric(
    aeroportos["longitude_deg"],
    errors="coerce",
)

aeroportos = aeroportos.loc[
    aeroportos["Latitude"].between(-90, 90)
    & aeroportos["Longitude"].between(-180, 180)
].copy()

# Associar os nomes dos países.
aeroportos = aeroportos.merge(
    paises,
    left_on="iso_country",
    right_on="code",
    how="left",
    suffixes=("_aeroporto", "_pais"),
)

aeroportos = aeroportos.rename(
    columns={
        "name_aeroporto": "Nome Aeroporto",
        "name_pais": "País",
        "municipality": "Cidade",
    }
)

colunas_geograficas = [
    "Codigo ICAO",
    "Nome Aeroporto",
    "Cidade",
    "País",
    "Latitude",
    "Longitude",
]

colunas_geograficas_ausentes = [
    coluna
    for coluna in colunas_geograficas
    if coluna not in aeroportos.columns
]

if colunas_geograficas_ausentes:
    raise ValueError(
        "Colunas geográficas ausentes: "
        f"{colunas_geograficas_ausentes}"
    )

aeroportos = (
    aeroportos[colunas_geograficas]
    .drop_duplicates(subset="Codigo ICAO", keep="first")
    .copy()
)

# 4. ASSOCIAR COORDENADAS ÀS ROTAS

dados_origem = aeroportos.rename(
    columns={
        "Codigo ICAO": col_origem,
        "Nome Aeroporto": "Nome Origem",
        "Cidade": "Cidade Origem",
        "País": "País Origem",
        "Latitude": "Lat Origem",
        "Longitude": "Lon Origem",
    }
)

dados_destino = aeroportos.rename(
    columns={
        "Codigo ICAO": col_destino,
        "Nome Aeroporto": "Nome Destino",
        "Cidade": "Cidade Destino",
        "País": "País Destino",
        "Latitude": "Lat Destino",
        "Longitude": "Lon Destino",
    }
)

rotas_mapa = resumo_rotas_mapa.merge(
    dados_origem,
    on=col_origem,
    how="inner",
)

rotas_mapa = rotas_mapa.merge(
    dados_destino,
    on=col_destino,
    how="inner",
)

rotas_mapa = rotas_mapa.dropna(
    subset=[
        "Lat Origem",
        "Lon Origem",
        "Lat Destino",
        "Lon Destino",
        "Percentual_Atrasado",
    ]
).copy()

if rotas_mapa.empty:
    raise ValueError(
        "Nenhuma rota pôde ser localizada geograficamente. "
        "Confira os códigos ICAO dos aeroportos."
    )

# 5. SELECIONAR AS ROTAS MAIS RELEVANTES

TOP_ROTAS = 20

rotas_mapa = (
    rotas_mapa
    .sort_values(
        ["Percentual_Atrasado", "Voos_Classificados"],
        ascending=[False, False],
    )
    .head(TOP_ROTAS)
    .reset_index(drop=True)
)

if rotas_mapa.empty:
    raise ValueError("Não existem rotas disponíveis para o mapa.")

# Limites reais dos percentuais que serão exibidos.
percentual_minimo = float(
    rotas_mapa["Percentual_Atrasado"].min()
)
percentual_maximo = float(
    rotas_mapa["Percentual_Atrasado"].max()
)

# Evitar divisão por zero caso todos os percentuais sejam iguais.
amplitude_percentual = percentual_maximo - percentual_minimo

if amplitude_percentual == 0:
    amplitude_percentual = 1.0

print("\n========== DIAGNÓSTICO DO MAPA ==========")
print(f"Rotas selecionadas: {len(rotas_mapa)}")
print(f"Mínimo de partidas classificadas: {MINIMO_VOOS_ROTA}")
print(f"Menor percentual exibido: {percentual_minimo:.1f}%")
print(f"Maior percentual exibido: {percentual_maximo:.1f}%")
print("=========================================\n")


# 6. DEFINIR A ESCALA VISUAL

from plotly.colors import sample_colorscale

escala_atrasos = [
    [0.00, "#9BC2E6"],  # Azul-claro: menor percentual
    [0.45, "#4F8FC5"],  # Azul médio
    [0.72, "#E6A15A"],  # Laranja: percentual elevado
    [1.00, "#D9534F"],  # Vermelho: maior percentual
]

def cor_por_percentual(percentual):
    """Retorna uma cor conforme a posição do percentual na amostra."""

    valor_normalizado = (
        float(percentual) - percentual_minimo
    ) / amplitude_percentual

    valor_normalizado = float(
        np.clip(valor_normalizado, 0, 1)
    )

    return sample_colorscale(
        escala_atrasos,
        [valor_normalizado],
    )[0]

# 7. CONSTRUIR O MAPA DE ROTAS

fig_mapa_atrasos = go.Figure()

for _, rota in rotas_mapa.iterrows():

    percentual = float(rota["Percentual_Atrasado"])
    cor_rota = cor_por_percentual(percentual)

    texto_hover = (
        f"<b>{rota['Nome Origem']} ({rota[col_origem]})"
        f" → {rota['Nome Destino']} ({rota[col_destino]})</b>"
        f"<br><br>"
        f"<b>Partidas atrasadas: {percentual:.1f}%</b>"
        f"<br>Partidas atrasadas: {rota['Voos_Atrasados']:,.0f}"
        f"<br>Partidas classificadas: "
        f"{rota['Voos_Classificados']:,.0f}"
        f"<br>Total de voos realizados: {rota['Total_Voos']:,.0f}"
        f"<br><br>"
        f"Origem: {rota['Cidade Origem']}, {rota['País Origem']}"
        f"<br>Destino: {rota['Cidade Destino']}, {rota['País Destino']}"
    )

    fig_mapa_atrasos.add_trace(
        go.Scattergeo(
            lon=[
                rota["Lon Origem"],
                rota["Lon Destino"],
                None,
            ],
            lat=[
                rota["Lat Origem"],
                rota["Lat Destino"],
                None,
            ],
            mode="lines",
            line=dict(
                color=cor_rota,
                width=2.8,
            ),
            text=[texto_hover, texto_hover, ""],
            hovertemplate="%{text}<extra></extra>",
            showlegend=False,
            name="Rota aérea",
        )
    )

# 8. ADICIONAR A ESCALA DE CORES

valores_escala = (
    np.linspace(percentual_minimo, percentual_maximo, 5)
    if percentual_maximo > percentual_minimo
    else [percentual_minimo]
)

fig_mapa_atrasos.add_trace(
    go.Scattergeo(
        lon=[None],
        lat=[None],
        mode="markers",
        marker=dict(
            size=1,
            color=[percentual_minimo],
            colorscale=escala_atrasos,
            cmin=percentual_minimo,
            cmax=(
                percentual_maximo
                if percentual_maximo > percentual_minimo
                else percentual_minimo + 1
            ),
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="% de partidas<br>atrasadas",
                    side="right",
                ),
                tickvals=valores_escala,
                ticktext=[
                    f"{valor:.1f}%"
                    for valor in valores_escala
                ],
                thickness=14,
                len=0.62,
                x=0.98,
                outlinewidth=0,
                tickfont=dict(size=10),
            ),
        ),
        hoverinfo="skip",
        showlegend=False,
        name="Escala de atrasos",
    )
)

# 9. MARCAR OS AEROPORTOS

pontos_origem = rotas_mapa[
    [
        col_origem,
        "Nome Origem",
        "Cidade Origem",
        "País Origem",
        "Lat Origem",
        "Lon Origem",
    ]
].rename(
    columns={
        col_origem: "ICAO",
        "Nome Origem": "Aeroporto",
        "Cidade Origem": "Cidade",
        "País Origem": "País",
        "Lat Origem": "Latitude",
        "Lon Origem": "Longitude",
    }
)

pontos_destino = rotas_mapa[
    [
        col_destino,
        "Nome Destino",
        "Cidade Destino",
        "País Destino",
        "Lat Destino",
        "Lon Destino",
    ]
].rename(
    columns={
        col_destino: "ICAO",
        "Nome Destino": "Aeroporto",
        "Cidade Destino": "Cidade",
        "País Destino": "País",
        "Lat Destino": "Latitude",
        "Lon Destino": "Longitude",
    }
)

pontos_aeroportos = (
    pd.concat(
        [pontos_origem, pontos_destino],
        ignore_index=True,
    )
    .drop_duplicates(subset="ICAO")
    .copy()
)

fig_mapa_atrasos.add_trace(
    go.Scattergeo(
        lon=pontos_aeroportos["Longitude"],
        lat=pontos_aeroportos["Latitude"],
        mode="markers",
        marker=dict(
            size=7,
            color="#174A7E",
            opacity=0.95,
            line=dict(
                color="#FFFFFF",
                width=1.2,
            ),
        ),
        text=[
            (
                f"<b>{linha['Aeroporto']}</b>"
                f"<br>ICAO: {linha['ICAO']}"
                f"<br>Cidade: {linha['Cidade']}"
                f"<br>País: {linha['País']}"
            )
            for _, linha in pontos_aeroportos.iterrows()
        ],
        hovertemplate="%{text}<extra></extra>",
        name="Aeroportos",
    )
)

# 10. CONFIGURAR O LAYOUT FINAL

aplicar_layout(fig_mapa_atrasos)

fig_mapa_atrasos.update_layout(
    title=dict(
        text=(
            "<b>Rotas com Maior Percentual de Partidas Atrasadas</b>"
            "<br>"
            "<span style='font-size:13px; font-weight:normal;'>"
            f"Top {len(rotas_mapa)} rotas | Voos realizados em 2025"
            f" | Mínimo de {MINIMO_VOOS_ROTA} partidas classificadas"
            "</span>"
        ),
        x=0.02,
        xanchor="left",
        y=0.94,
        yanchor="top",
        font=dict(
            family="Arial",
            size=20,
            color="#174A7E",
        ),
    ),
    template="plotly_white",
    height=700,
    margin=dict(l=15, r=90, t=125, b=25),
    paper_bgcolor="#FFFFFF",
    plot_bgcolor="#FFFFFF",
    showlegend=False,
    hoverlabel=dict(
        bgcolor="#FFFFFF",
        bordercolor="#174A7E",
        font=dict(
            family="Arial",
            size=12,
            color="#263445",
        ),
        align="left",
    ),
    geo=dict(
        projection=dict(type="natural earth"),
        showland=True,
        landcolor="#F2F5F9",
        showocean=True,
        oceancolor="#FFFFFF",
        showcountries=True,
        countrycolor="#CCD5E0",
        countrywidth=0.7,
        showcoastlines=True,
        coastlinecolor="#AAB7C5",
        coastlinewidth=0.7,
        showframe=False,
        bgcolor="#FFFFFF",
        lataxis=dict(
            showgrid=True,
            gridcolor="#E8EEF5",
            gridwidth=0.5,
        ),
        lonaxis=dict(
            showgrid=True,
            gridcolor="#E8EEF5",
            gridwidth=0.5,
        ),
    ),
)

# 11. PREPARAR O HTML PARA O DASHBOARD

html_mapa_atrasos = pio.to_html(
    fig_mapa_atrasos,
    full_html=False,
    include_plotlyjs=False,
    config={
        "responsive": True,
        "displayModeBar": True,
        "displaylogo": False,
        "scrollZoom": True,
    },
)

print("\nMapa de atrasos preparado para o dashboard.")
print(f"Rotas exibidas: {len(rotas_mapa)}")
print(f"Aeroportos identificados: {len(pontos_aeroportos)}")
print(f"Percentual mínimo: {percentual_minimo:.1f}%")
print(f"Percentual máximo: {percentual_maximo:.1f}%")

######### ETAPA 5.4.6 — DASHBOARD DE ATRASOS #########

# 1. PREPARAR BASE ANALÍTICA

df_atrasos = df_pontualidade.copy()

df_atrasos["Grupo Pontualidade"] = (
    df_atrasos["Classificação Partida"]
    .apply(agrupar_pontualidade)
)

df_atrasos = df_atrasos.loc[
    df_atrasos["Grupo Pontualidade"].isin(
        [
            "Antecipado",
            "Pontual",
            "Atrasado"
        ]
    )
].copy()

df_atrasos["Rota"] = (
    df_atrasos["Sigla ICAO Aeroporto Origem"]
    .fillna("Não informado")
    .astype(str)
    + " → "
    +
    df_atrasos["Sigla ICAO Aeroporto Destino"]
    .fillna("Não informado")
    .astype(str)
)

df_atrasos["Aeroporto de origem"] = (
    df_atrasos["Sigla ICAO Aeroporto Origem"]
    .fillna("Não informado")
    .astype(str)
    + " — "
    +
    df_atrasos["Descrição Aeroporto Origem"]
    .fillna("Descrição não informada")
    .astype(str)
)

# 2. FUNÇÃO DE RESUMO

def resumir_atrasos(
    base,
    coluna_grupo,
    nome_analise,
    minimo_voos
):

    resumo = (
        base
        .groupby(
            [
                coluna_grupo,
                "Grupo Pontualidade"
            ]
        )
        .size()
        .unstack(fill_value=0)
    )

    for categoria in [
        "Antecipado",
        "Pontual",
        "Atrasado"
    ]:

        if categoria not in resumo.columns:
            resumo[categoria] = 0

    resumo["Total_Classificado"] = (
        resumo["Antecipado"]
        + resumo["Pontual"]
        + resumo["Atrasado"]
    )

    resumo["Total_Atrasado"] = (
        resumo["Atrasado"]
    )

    resumo = resumo.loc[
        resumo["Total_Classificado"]
        >= minimo_voos
    ].copy()

    resumo["Percentual_Atrasado"] = (
        resumo["Total_Atrasado"]
        .div(resumo["Total_Classificado"])
        .mul(100)
    )

    resumo = (
        resumo
        .reset_index()
        .rename(
            columns={
                coluna_grupo: "Grupo"
            }
        )
    )

    resumo["Analise"] = nome_analise

    return resumo[
        [
            "Analise",
            "Grupo",
            "Percentual_Atrasado",
            "Total_Classificado",
            "Total_Atrasado"
        ]
    ]

# 3. GERAR RANKINGS

resumo_empresa = resumir_atrasos(
    base=df_atrasos,
    coluna_grupo="Empresa Aérea",
    nome_analise="Companhias aéreas",
    minimo_voos=1000
)

resumo_rota = resumir_atrasos(
    base=df_atrasos,
    coluna_grupo="Rota",
    nome_analise="Rotas",
    minimo_voos=500
)

resumo_aeroporto = resumir_atrasos(
    base=df_atrasos,
    coluna_grupo="Aeroporto de origem",
    nome_analise="Aeroportos de origem",
    minimo_voos=1000
)

# 4. TOP 15 — RANKINGS COM FILTRO INTERATIVO

ranking_empresa = (
    resumo_empresa.nlargest(15, "Percentual_Atrasado")
    .sort_values("Percentual_Atrasado", ascending=True)
)
ranking_rota = (
    resumo_rota.nlargest(15, "Percentual_Atrasado")
    .sort_values("Percentual_Atrasado", ascending=True)
)
ranking_aeroporto = (
    resumo_aeroporto.nlargest(15, "Percentual_Atrasado")
    .sort_values("Percentual_Atrasado", ascending=True)
)

# 5. GRÁFICO INTERATIVO DE RANKINGS

datasets = [
    (
        ranking_empresa,
        "Companhias aéreas",
        "Companhias com maior percentual de partidas atrasadas",
        [
            [0.00, "#C7DDF0"],
            [0.50, "#78A6D4"],
            [1.00, "#174A7E"]
        ]
    ),
    (
        ranking_rota,
        "Rotas",
        "Rotas com maior percentual de partidas atrasadas",
        [
            [0.00, "#F8D9B8"],
            [0.50, "#E6A15A"],
            [1.00, "#C66A1F"]
        ]
    ),
    (
        ranking_aeroporto,
        "Aeroportos de origem",
        "Aeroportos com maior percentual de partidas atrasadas",
        [
            [0.00, "#BDE3C9"],
            [0.50, "#63B27C"],
            [1.00, "#2E8B57"]
        ]
    )
]

fig_atrasos_dashboard = go.Figure()

medias_grupo = []

for i, (dados, nome_grupo, titulo, escala_cor) in enumerate(datasets):

    media_grupo = dados["Percentual_Atrasado"].mean()
    medias_grupo.append(media_grupo)

    fig_atrasos_dashboard.add_trace(

        go.Bar(

            x=dados["Percentual_Atrasado"],

            y=dados["Grupo"],

            orientation="h",

            visible=(i == 0),

            marker=dict(
                color=dados["Percentual_Atrasado"],
                colorscale=escala_cor,
                line=dict(
                    color="white",
                    width=1
                ),
                showscale=False
            ),

            text=[
                (
                    f"{taxa:.1f}%"
                    f"<br>{volume:,.0f} voos"
                ).replace(",", ".")
                for taxa, volume in zip(
                    dados["Percentual_Atrasado"],
                    dados["Total_Classificado"]
                )
            ],

            textposition="outside",

            textfont=dict(
                size=10,
                color="#24364B"
            ),

            cliponaxis=False,

            customdata=np.column_stack(
                [
                    dados["Total_Classificado"],
                    dados["Total_Atrasado"]
                ]
            ),

            hovertemplate=(
                "<b>%{y}</b><br><br>"
                "Percentual atrasado: <b>%{x:.2f}%</b><br>"
                "Voos classificados: %{customdata[0]:,.0f}<br>"
                "Voos atrasados: %{customdata[1]:,.0f}"
                "<extra></extra>"
            ),

            name=nome_grupo
        )
    )

botoes_ranking = []

for i, (_, nome_grupo, titulo, _) in enumerate(datasets):

    visibilidade = [False] * len(datasets)
    visibilidade[i] = True

    botoes_ranking.append(

        dict(
            label=nome_grupo,

            method="update",

            args=[
                {
                    "visible": visibilidade
                },

                {
                    "title.text": (
                        f"<b>Ranking de atrasos por "
                        f"{nome_grupo.lower()}</b>"
                        f"<br><sup>{titulo} • Top 15</sup>"
                    ),

                    "shapes": [
                        dict(
                            type="line",
                            x0=medias_grupo[i],
                            x1=medias_grupo[i],
                            y0=0,
                            y1=1,
                            yref="paper",
                            line=dict(
                                color="#D9534F",
                                width=2,
                                dash="dash"
                            )
                        )
                    ],

                    "annotations": [

                        dict(
                            text=f"Média: {medias_grupo[i]:.1f}%",
                            x=medias_grupo[i],
                            y=1.02,
                            yref="paper",
                            showarrow=False,
                            font=dict(
                                size=11,
                                color="#D9534F"
                            )
                        ),

                        dict(
                            text="Selecione o grupo que deseja analisar",
                            x=0.98,
                            y=1.19,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            xanchor="right",
                            font=dict(
                                size=11,
                                color="#64748B"
                            )
                        ),

                        dict(
                            text=(
                                "Percentuais calculados entre os voos "
                                "com classificação disponível; "
                                "o volume mínimo varia conforme o grupo."
                            ),
                            x=0.00,
                            y=-0.13,
                            xref="paper",
                            yref="paper",
                            showarrow=False,
                            xanchor="left",
                            font=dict(
                                size=10,
                                color="#64748B"
                            )
                        )
                    ]
                }
            ]
        )
    )

fig_atrasos_dashboard.update_layout(

    title=dict(
        text=(
            "<b>Ranking de atrasos por companhias aéreas</b>"
            "<br><sup>"
            "Companhias com maior percentual de partidas atrasadas • Top 15"
            "</sup>"
        ),

        x=0.02,
        xanchor="left",

        y=0.94,
        yanchor="top",

        font=dict(
            size=21,
            color="#1B263B"
        ),
    ),

    template="plotly_white",

    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",

    autosize=True,

    height=720,

    font=dict(
        family="Segoe UI",
        size=12,
        color="#24364B"
    ),

    margin=dict(
        l=245,
        r=90,
        t=175,
        b=75
    ),

    showlegend=False,

    bargap=0.30,

    updatemenus=[
        dict(
            type="dropdown",

            buttons=botoes_ranking,

            direction="down",

            showactive=True,

            x=0.98,
            y=1.08,

            xanchor="right",
            yanchor="top",

            bgcolor="white",
            bordercolor="#CBD5E1",
            borderwidth=1,

            font=dict(
                size=12,
                color="#24364B"
            ),
        )
    ],

    shapes=[
        dict(
            type="line",
            x0=medias_grupo[0],
            x1=medias_grupo[0],
            y0=0,
            y1=1,
            yref="paper",
            line=dict(
                color="#D9534F",
                width=2,
                dash="dash"
            )
        )
    ],

    annotations=[

        dict(
            text=f"Média: {medias_grupo[0]:.1f}%",
            x=medias_grupo[0],
            y=1.02,
            yref="paper",
            showarrow=False,
            font=dict(
                size=11,
                color="#D9534F"
            )
        ),

        dict(
            text="Selecione o grupo que deseja analisar",
            x=0.98,
            y=1.19,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="right",
            font=dict(
                size=11,
                color="#64748B"
            )
        ),

        dict(
            text=(
                "Percentuais calculados entre os voos com classificação "
                "disponível; o volume mínimo varia conforme o grupo."
            ),
            x=0.00,
            y=-0.13,
            xref="paper",
            yref="paper",
            showarrow=False,
            xanchor="left",
            font=dict(
                size=10,
                color="#64748B"
            )
        )
    ],

    hoverlabel=dict(
        bgcolor="white",
        font_size=12,
        font_family="Segoe UI"
    )
)

fig_atrasos_dashboard.update_xaxes(
    title="Percentual de partidas atrasadas (%)",
    ticksuffix="%",
    rangemode="tozero",
    showgrid=True,
    gridwidth=1,
    gridcolor="#E8EDF2",
    zeroline=False,
    showline=True,
    linewidth=1,
    linecolor="#D9E3F0"
)

fig_atrasos_dashboard.update_yaxes(
    title=None,
    automargin=True,
    autorange="reversed",
    showgrid=False,
    showline=False
)

aplicar_layout(fig_atrasos_dashboard)

######### ETAPA 5.4.7 — QUADRANTE DE RISCO OPERACIONAL DOS ATRASOS #########

risco_atrasos = resumo_empresa.copy()

risco_atrasos["Impacto"] = (
    risco_atrasos["Total_Atrasado"]
)

media_volume = (
    risco_atrasos["Total_Classificado"]
    .mean()
)

media_atraso = (
    risco_atrasos["Percentual_Atrasado"]
    .mean()
)

fig_quadrante_atrasos = px.scatter(

    risco_atrasos,

    x="Total_Classificado",

    y="Percentual_Atrasado",

    size="Impacto",

    color="Percentual_Atrasado",

    hover_name="Grupo",

    color_continuous_scale=[
        [0.00, "#C7DDF0"],
        [0.50, "#78A6D4"],
        [0.75, "#E6A15A"],
        [1.00, "#D9534F"]
    ],

    labels={
        "Total_Classificado": "Voos classificados",
        "Percentual_Atrasado": "% de atraso"
    }
)

fig_quadrante_atrasos.update_traces(

    marker=dict(
        line=dict(
            color="white",
            width=1
        ),
        opacity=0.85
    ),

    hovertemplate=
    (
        "<b>%{hovertext}</b><br><br>"
        "Percentual atrasado: %{y:.2f}%<br>"
        "Voos classificados: %{x:,.0f}<br>"
        "Voos atrasados: %{marker.size:,.0f}"
        "<extra></extra>"
    )
)

fig_quadrante_atrasos.add_vline(

    x=media_volume,

    line_dash="dash",

    line_color="#64748B",

    opacity=0.8
)

fig_quadrante_atrasos.add_hline(

    y=media_atraso,

    line_dash="dash",

    line_color="#64748B",

    opacity=0.8
)

fig_quadrante_atrasos.add_annotation(

    x=media_volume * 1.02,

    y=media_atraso * 1.02,

    text="Média do grupo",

    showarrow=False,

    font=dict(
        size=11,
        color="#64748B"
    )
)

fig_quadrante_atrasos.update_layout(

    title=dict(
        text=(
            "<b>Quadrante de risco operacional dos atrasos</b>"
            "<br><sup>"
            "Volume de voos × taxa de atraso × impacto operacional"
            "</sup>"
        ),

        x=0.02,

        font=dict(
            size=21,
            color="#1B263B"
        )
    ),

    template="plotly_white",

    height=700,

    plot_bgcolor="white",

    paper_bgcolor="white",

    coloraxis_colorbar=dict(
        title="% atraso"
    ),

    margin=dict(
        l=60,
        r=40,
        t=140,
        b=60
    ),

    font=dict(
        family="Segoe UI",
        size=12,
        color="#24364B"
    ),

    annotations=[

        dict(
            text="CRÍTICO",
            xref="paper",
            yref="paper",
            x=0.85,
            y=0.90,
            showarrow=False,
            font=dict(
                size=14,
                color="#D9534F"
            )
        ),

        dict(
            text="BAIXO RISCO",
            xref="paper",
            yref="paper",
            x=0.15,
            y=0.15,
            showarrow=False,
            font=dict(
                size=14,
                color="#2E8B57"
            )
        )
    ]
)

fig_quadrante_atrasos.update_xaxes(

    title="Volume de voos classificados",

    showgrid=True,

    gridcolor="#E8EDF2",

    showline=True,

    linecolor="#D9E3F0"
)

fig_quadrante_atrasos.update_yaxes(

    title="Percentual de partidas atrasadas (%)",

    ticksuffix="%",

    showgrid=True,

    gridcolor="#E8EDF2",

    showline=True,

    linecolor="#D9E3F0"
)

aplicar_layout(fig_quadrante_atrasos)

######### ETAPA 5.5 — ANÁLISE DAS DIFERENÇAS DE HORÁRIO #########

df_atrasos = df_voos_realizados.copy()

print("\nRESUMO DAS DIFERENÇAS DE PARTIDA (MINUTOS)")
print(df_atrasos["Diferença Partida (min)"].describe().round(2))

print("\nRESUMO DAS DIFERENÇAS DE CHEGADA (MINUTOS)")
print(df_atrasos["Diferença Chegada (min)"].describe().round(2))

for coluna in colunas_diferencas:
    valores = df_atrasos[coluna].dropna()

    print(f"\n{coluna.upper()}")
    print(f"Diferenças calculáveis: {len(valores):,}")
    print(
        "Diferenças ausentes:",
        f"{int(df_atrasos[coluna].isna().sum()):,}"
    )

    print(f"Antecipações: {int(valores.lt(0).sum()):,}")
    print(f"Horários iguais: {int(valores.eq(0).sum()):,}")
    print(f"Atrasos: {int(valores.gt(0).sum()):,}")

    if len(valores) > 0:
        print(
            "Antecipações entre avaliáveis: "
            f"{valores.lt(0).mean() * 100:.2f}%"
        )

        print(
            "Horários iguais entre avaliáveis: "
            f"{valores.eq(0).mean() * 100:.2f}%"
        )

        print(
            "Atrasos entre avaliáveis: "
            f"{valores.gt(0).mean() * 100:.2f}%"
        )

    print(
        "Ausentes sobre todos os voos realizados:",
        f"{df_atrasos[coluna].isna().mean() * 100:.2f}%"
    )

######### 5.5.1 — VALIDAÇÃO DAS CATEGORIAS DE DIFERENÇA #########

print("\nVALIDAÇÃO DAS DIFERENÇAS DE HORÁRIO")

for coluna in colunas_diferencas:

    diferencas = df_atrasos[coluna]

    antecipacoes = int(diferencas.lt(0).sum())
    horarios_iguais = int(diferencas.eq(0).sum())
    atrasos = int(diferencas.gt(0).sum())
    ausentes = int(diferencas.isna().sum())
    calculaveis = int(diferencas.notna().sum())

    soma_categorias = (
        antecipacoes + horarios_iguais + atrasos + ausentes
    )

    print(f"\n{coluna}")
    print(f"Diferenças calculáveis: {calculaveis:,}")
    print(f"Antecipações: {antecipacoes:,}")
    print(f"Horários iguais: {horarios_iguais:,}")
    print(f"Atrasos: {atrasos:,}")
    print(f"Diferenças ausentes: {ausentes:,}")
    print(f"Soma das categorias: {soma_categorias:,}")

    assert soma_categorias == len(df_atrasos), (
        f"As categorias de '{coluna}' não fecham com "
        "o total de voos realizados."
    )

    assert calculaveis == 949_891, (
        f"A quantidade de diferenças calculáveis em '{coluna}' "
        "está diferente do esperado. Confira a Etapa 4."
    )

print("\nValidação das diferenças concluída.")


######### ETAPA 5.5.2 — MAIORES ATRASOS E ANTECIPAÇÕES #########

colunas_exibicao = [
    "Empresa Aérea",
    "Número Voo",
    "Sigla ICAO Aeroporto Origem",
    "Sigla ICAO Aeroporto Destino",
    "Partida Prevista",
    "Partida Real",
    "Diferença Partida (min)",
    "Chegada Prevista",
    "Chegada Real",
    "Diferença Chegada (min)"
]

colunas_exibicao_ausentes = [
    coluna for coluna in colunas_exibicao
    if coluna not in df_atrasos.columns
]

if colunas_exibicao_ausentes:
    raise KeyError(
        "Colunas ausentes para investigação: "
        + ", ".join(colunas_exibicao_ausentes)
    )

investigacoes_extremos = [
    (
        "MAIORES DIFERENÇAS POSITIVAS DE PARTIDA",
        "Diferença Partida (min)",
        True
    ),
    (
        "MAIORES ANTECIPAÇÕES DE PARTIDA",
        "Diferença Partida (min)",
        False
    ),
    (
        "MAIORES DIFERENÇAS POSITIVAS DE CHEGADA",
        "Diferença Chegada (min)",
        True
    ),
    (
        "MAIORES ANTECIPAÇÕES DE CHEGADA",
        "Diferença Chegada (min)",
        False
    )
]

for titulo, coluna, maiores in investigacoes_extremos:

    print(f"\n{titulo}")

    dados_validos = df_atrasos.loc[df_atrasos[coluna].notna()]

    if maiores:
        resultado = dados_validos.nlargest(10, coluna)
    else:
        resultado = dados_validos.nsmallest(10, coluna)

    if resultado.empty:
        print("Nenhum registro disponível para esta análise.")
    else:
        print(resultado[colunas_exibicao].to_string(index=False))


######### ETAPA 5.5.3 — INVESTIGAÇÃO DE DIFERENÇAS EXTREMAS #########

dif_partida = df_atrasos["Diferença Partida (min)"]
dif_chegada = df_atrasos["Diferença Chegada (min)"]

extremo_partida = (
    dif_partida.abs().gt(limite_extremo).fillna(False)
)

extremo_chegada = (
    dif_chegada.abs().gt(limite_extremo).fillna(False)
)

mask_extremos = extremo_partida | extremo_chegada

print("\nINVESTIGAÇÃO DE DIFERENÇAS SUPERIORES A 24 HORAS")

print("Partidas extremas:", f"{int(extremo_partida.sum()):,}")
print("Chegadas extremas:", f"{int(extremo_chegada.sum()):,}")
print(
    "Registros extremos em partida ou chegada:",
    f"{int(mask_extremos.sum()):,}"
)
print(
    "Registros extremos nas duas etapas:",
    f"{int((extremo_partida & extremo_chegada).sum()):,}"
)

print("\nPARTIDAS — DIFERENÇAS EXTREMAS")
print(
    "Mais de 24 horas depois:",
    f"{int(dif_partida.gt(limite_extremo).sum()):,}"
)
print(
    "Mais de 24 horas antes:",
    f"{int(dif_partida.lt(-limite_extremo).sum()):,}"
)

print("\nCHEGADAS — DIFERENÇAS EXTREMAS")
print(
    "Mais de 24 horas depois:",
    f"{int(dif_chegada.gt(limite_extremo).sum()):,}"
)
print(
    "Mais de 24 horas antes:",
    f"{int(dif_chegada.lt(-limite_extremo).sum()):,}"
)

df_extremos = df_atrasos.loc[mask_extremos].copy()

# Criar as colunas mesmo quando df_extremos estiver vazio.
df_extremos["Maior diferença absoluta (min)"] = (
    df_extremos[colunas_diferencas].abs().max(axis=1)
)

df_extremos["Rota"] = (
    df_extremos["Sigla ICAO Aeroporto Origem"]
    .astype("string")
    .fillna("Não informado")
    + " → "
    + df_extremos["Sigla ICAO Aeroporto Destino"]
    .astype("string")
    .fillna("Não informado")
)

print("\n15 REGISTROS COM MAIORES DIFERENÇAS ABSOLUTAS")

if df_extremos.empty:
    print("Nenhum registro com diferença absoluta superior a 24 horas.")
else:
    print(
        df_extremos.nlargest(
            15, "Maior diferença absoluta (min)"
        )[
            colunas_exibicao + ["Maior diferença absoluta (min)"]
        ].to_string(index=False)
    )

######### ETAPA 5.5.4 — CLASSIFICAÇÕES ANAC NOS REGISTROS EXTREMOS #########

df_atrasos["Classificação Partida Análise"] = (
    padronizar_categoria(df_atrasos["Situação Partida"])
)

df_atrasos["Classificação Chegada Análise"] = (
    padronizar_categoria(df_atrasos["Situação Chegada"])
)

print("\nCLASSIFICAÇÃO DAS PARTIDAS — DIFERENÇAS EXTREMAS")
print(
    df_atrasos.loc[extremo_partida, "Classificação Partida Análise"]
    .fillna("Sem classificação")
    .value_counts()
    .to_string()
)

print("\nCLASSIFICAÇÃO DAS CHEGADAS — DIFERENÇAS EXTREMAS")
print(
    df_atrasos.loc[extremo_chegada, "Classificação Chegada Análise"]
    .fillna("Sem classificação")
    .value_counts()
    .to_string()
)

colunas_validacao = [
    "Empresa Aérea",
    "Número Voo",
    "Partida Prevista",
    "Partida Real",
    "Diferença Partida (min)",
    "Situação Partida",
    "Chegada Prevista",
    "Chegada Real",
    "Diferença Chegada (min)",
    "Situação Chegada"
]

print("\nEXEMPLOS DE REGISTROS COM DIFERENÇAS EXTREMAS")

if df_extremos.empty:
    print("Nenhum registro para exibir.")
else:
    print(
        df_atrasos.loc[mask_extremos, colunas_validacao]
        .head(20)
        .to_string(index=False)
    )

######### ETAPA 5.5.5 — DISTRIBUIÇÃO CUMULATIVA DOS ATRASOS #########

limites_atraso = [60, 120, 240, 720, 1440, 2880]

for coluna, titulo in [
    ("Diferença Partida (min)", "PARTIDA"),
    ("Diferença Chegada (min)", "CHEGADA")
]:

    print(f"\nATRASOS DE {titulo} POR LIMITE CUMULATIVO")

    valores = df_atrasos[coluna].dropna()
    for limite in limites_atraso:
        quantidade = int(valores.gt(limite).sum())
        if len(valores) > 0:
            percentual = quantidade / len(valores) * 100
            print(
                f"Superior a {limite} minutos "
                f"({limite / 60:g} horas): "
                f"{quantidade:,} ({percentual:.2f}% dos avaliáveis)"
            )
        else:
            print(
                f"Superior a {limite} minutos: "
                "sem diferenças calculáveis."
            )

######### ETAPA 5.5.6 — COMPARAÇÃO COM A CLASSIFICAÇÃO DA ANAC #########

for etapa, coluna_diferenca, coluna_situacao in [
    (
        "PARTIDA",
        "Diferença Partida (min)",
        "Classificação Partida Análise"
    ),
    (
        "CHEGADA",
        "Diferença Chegada (min)",
        "Classificação Chegada Análise"
    )
]:

    diferenca = df_atrasos[coluna_diferenca]
    situacao = df_atrasos[coluna_situacao]

    calculada = diferenca.gt(240).fillna(False)
    anac = situacao.eq("Atraso > 240").fillna(False)

    avaliavel = diferenca.notna()
    classificacao_preenchida = situacao.notna()

    print(f"\nVALIDAÇÃO DOS ATRASOS DE {etapa}")

    print("Diferenças calculáveis:", f"{int(avaliavel.sum()):,}")

    print(
        "Diferenças calculadas acima de 240 min:",
        f"{int((calculada & avaliavel).sum()):,}"
    )

    print(
        "Classificados pela ANAC como Atraso > 240:",
        f"{int((anac & avaliavel).sum()):,}"
    )

    print(
        "Acima de 240 min com outra classificação preenchida:",
        f"{int((calculada & classificacao_preenchida & ~anac).sum()):,}"
    )

    print(
        "Acima de 240 min sem classificação:",
        f"{int((calculada & ~classificacao_preenchida).sum()):,}"
    )

    print(
        "Classificados pela ANAC como > 240 min, mas diferença "
        "calculada não superior a 240 min:",
        f"{int((avaliavel & ~calculada & anac).sum()):,}"
    )

######### ETAPA 5.5.7 — INVESTIGAÇÃO DAS DIVERGÊNCIAS #########

situacao_partida_analise = (
    df_atrasos["Classificação Partida Análise"]
)

situacao_chegada_analise = (
    df_atrasos["Classificação Chegada Análise"]
)

divergencias_partida_mask = (
    df_atrasos["Diferença Partida (min)"].gt(240)
    & situacao_partida_analise.notna()
    & situacao_partida_analise.ne("Atraso > 240")
).fillna(False)

divergencias_chegada_mask = (
    df_atrasos["Diferença Chegada (min)"].gt(240)
    & situacao_chegada_analise.notna()
    & situacao_chegada_analise.ne("Atraso > 240")
).fillna(False)

divergencias_partida = df_atrasos.loc[
    divergencias_partida_mask
].copy()

divergencias_chegada = df_atrasos.loc[
    divergencias_chegada_mask
].copy()

print("\nDIVERGÊNCIAS NA CLASSIFICAÇÃO DAS PARTIDAS")
print(f"Quantidade: {len(divergencias_partida):,}")

if divergencias_partida.empty:
    print("Nenhuma divergência identificada por este critério.")
else:
    print(
        divergencias_partida[
            [
                "Empresa Aérea",
                "Número Voo",
                "Sigla ICAO Aeroporto Origem",
                "Sigla ICAO Aeroporto Destino",
                "Partida Prevista",
                "Partida Real",
                "Diferença Partida (min)",
                "Situação Partida"
            ]
        ].to_string(index=False)
    )

print("\nDIVERGÊNCIAS NA CLASSIFICAÇÃO DAS CHEGADAS")
print(f"Quantidade: {len(divergencias_chegada):,}")

if divergencias_chegada.empty:
    print("Nenhuma divergência identificada por este critério.")
else:
    print(
        divergencias_chegada[
            [
                "Empresa Aérea",
                "Número Voo",
                "Sigla ICAO Aeroporto Origem",
                "Sigla ICAO Aeroporto Destino",
                "Chegada Prevista",
                "Chegada Real",
                "Diferença Chegada (min)",
                "Situação Chegada"
            ]
        ].to_string(index=False)
    )

######### ETAPA 5.5.8 — ANÁLISE DO LIMITE DE 240 MINUTOS #########

faixa_partida = df_atrasos.loc[
    df_atrasos["Diferença Partida (min)"].between(
        230, 250, inclusive="both"
    )
].copy()

faixa_chegada = df_atrasos.loc[
    df_atrasos["Diferença Chegada (min)"].between(
        230, 250, inclusive="both"
    )
].copy()

print("\nPARTIDAS — DIFERENÇAS ENTRE 230 E 250 MINUTOS")

if faixa_partida.empty:
    print("Nenhum registro encontrado nessa faixa.")
else:
    faixa_partida["Classificação Partida Análise"] = (
        faixa_partida["Classificação Partida Análise"]
        .fillna("Sem classificação")
    )

    print(
        pd.crosstab(
            faixa_partida["Diferença Partida (min)"],
            faixa_partida["Classificação Partida Análise"],
            dropna=False
        ).to_string()
    )

print("\nCHEGADAS — DIFERENÇAS ENTRE 230 E 250 MINUTOS")

if faixa_chegada.empty:
    print("Nenhum registro encontrado nessa faixa.")
else:
    faixa_chegada["Classificação Chegada Análise"] = (
        faixa_chegada["Classificação Chegada Análise"]
        .fillna("Sem classificação")
    )

    print(
        pd.crosstab(
            faixa_chegada["Diferença Chegada (min)"],
            faixa_chegada["Classificação Chegada Análise"],
            dropna=False
        ).to_string()
    )

######### ETAPA 5.5.9 — REGISTROS EXTREMOS POR EMPRESA E ROTA #########

print("\nREGISTROS EXTREMOS POR EMPRESA AÉREA")

if df_extremos.empty:
    print("Nenhum registro extremo para agrupar.")
else:
    print(
        df_extremos["Empresa Aérea"]
        .fillna("Empresa não informada")
        .value_counts()
        .head(15)
        .to_string()
    )

print("\nREGISTROS EXTREMOS POR ROTA")

if df_extremos.empty:
    print("Nenhuma rota extrema para exibir.")
else:
    print(
        df_extremos["Rota"]
        .value_counts()
        .head(15)
        .to_string()
    )

print("\nPADRÃO DAS DIFERENÇAS EXTREMAS")

if df_extremos.empty:
    print("Não há registros para comparar.")
else:
    print(
        pd.crosstab(
            df_extremos["Diferença Partida (min)"].abs().gt(limite_extremo),
            df_extremos["Diferença Chegada (min)"].abs().gt(limite_extremo),
            rownames=["Partida extrema"],
            colnames=["Chegada extrema"],
            dropna=False
        ).to_string()
    )

colunas_revisao = [
    "Empresa Aérea",
    "Número Voo",
    "Rota",
    "Partida Prevista",
    "Partida Real",
    "Diferença Partida (min)",
    "Chegada Prevista",
    "Chegada Real",
    "Diferença Chegada (min)",
    "Maior diferença absoluta (min)"
]

print("\nREGISTROS EXTREMOS PARA REVISÃO")

if df_extremos.empty:
    print("Nenhum registro para revisar.")
else:
    print(
        df_extremos.nlargest(
            20, "Maior diferença absoluta (min)"
        )[colunas_revisao].to_string(index=False)
    )

######### ETAPA 5.5.10 — VARIAÇÃO ENTRE PARTIDA E CHEGADA #########

# A coluna também é criada quando não existem registros extremos.
df_extremos["Variação entre diferenças (min)"] = (
    df_extremos["Diferença Chegada (min)"]
    - df_extremos["Diferença Partida (min)"]
)

variacao_valida = (
    df_extremos["Diferença Partida (min)"].notna()
    & df_extremos["Diferença Chegada (min)"].notna()
)

df_consistencia = df_extremos.loc[variacao_valida].copy()
print(
    "\nVARIAÇÃO ENTRE AS DIFERENÇAS DE PARTIDA E CHEGADA "
    "NOS REGISTROS EXTREMOS"
)

if df_consistencia.empty:
    print(
        "Não há registros extremos com ambas as diferenças calculáveis."
    )
else:
    print(
        df_consistencia["Variação entre diferenças (min)"]
        .describe()
        .round(2)
    )

    variacao_pequena = (
        df_consistencia["Variação entre diferenças (min)"]
        .abs()
        .le(30)
    )

    print(
        "Registros com variação absoluta de até 30 minutos:",
        f"{int(variacao_pequena.sum()):,}"
    )

    print(
        "Percentual entre os registros extremos avaliáveis:",
        f"{variacao_pequena.mean() * 100:.2f}%"
    )

    colunas_consistencia = [
        "Empresa Aérea",
        "Número Voo",
        "Sigla ICAO Aeroporto Origem",
        "Sigla ICAO Aeroporto Destino",
        "Partida Prevista",
        "Partida Real",
        "Diferença Partida (min)",
        "Chegada Prevista",
        "Chegada Real",
        "Diferença Chegada (min)",
        "Variação entre diferenças (min)"
    ]

    print("\nEXEMPLOS COM VARIAÇÃO ABSOLUTA DE ATÉ 30 MINUTOS")

    print(
        df_consistencia.loc[
            variacao_pequena, colunas_consistencia
        ].head(20).to_string(index=False)
    )

######### ETAPA 5.6 — RESUMO FINAL DA ANÁLISE EXPLORATÓRIA #########

print("\n" + "=" * 60)
print("RESUMO FINAL — ETAPA 5")
print("=" * 60)

print(f"Total de registros na base analítica: {len(df_eda):,}")
print("Total de voos realizados:", f"{int(mask_realizado.sum()):,}")
print("Total de voos cancelados:", f"{int(mask_cancelado.sum()):,}")

print(
    "Voos realizados contabilizados por mês da partida real:",
    f"{int(voos_por_mes.sum()):,}"
)

print("Voos sem partida real válida:", f"{voos_sem_partida_real:,}")

print(
    "Registros incluídos no cálculo da taxa de cancelamento:",
    f"{len(df_programados):,}"
)

print(
    "Registros sem partida prevista:",
    f"{len(registros_sem_partida_prevista):,}"
)

print(
    "Registros com situação inválida:",
    f"{registros_situacao_invalida:,}"
)

print(
    "Registros fora do cálculo da taxa:",
    f"{registros_sem_programacao_avaliavel:,}"
)

print("Registros extremos de partida:", f"{int(extremo_partida.sum()):,}")
print("Registros extremos de chegada:", f"{int(extremo_chegada.sum()):,}")
print(
    "Registros extremos em partida ou chegada:",
    f"{int(mask_extremos.sum()):,}"
)

print(
    "Divergências investigadas na classificação das partidas:",
    f"{len(divergencias_partida):,}"
)

print(
    "Divergências investigadas na classificação das chegadas:",
    f"{len(divergencias_chegada):,}"
)

print("=" * 60)

print(
    "\nNOTAS METODOLÓGICAS:"
    "\n1. A situação do voo define se o registro é REALIZADO ou CANCELADO."
    "\n2. O volume mensal de realizados utiliza o mês da Partida Real."
    "\n3. A comparação mensal e a taxa de cancelamento utilizam o mês "
    "da Partida Prevista."
    "\n4. Registros sem Partida Prevista ficam fora do cálculo mensal "
    "da taxa de cancelamento."
    "\n5. Os percentuais de classificação usam todos os realizados "
    "ou somente os classificados, conforme a análise."
    "\n6. Diferenças extremas e divergências são ocorrências para "
    "investigação, não erros automaticamente confirmados."
    "\n7. Médias de diferenças de horário podem ser influenciadas "
    "por antecipações e valores extremos; não representam, isoladamente, "
    "o atraso típico dos voos."
)

######### ETAPA 5.7 — VALIDAÇÃO FINAL DAS BASES E DOS RESULTADOS #########

print("\n" + "=" * 60)
print("VALIDAÇÃO FINAL — BASES DA ANÁLISE")
print("=" * 60)

# 1. Base tratada: duplicatas foram removidas na Etapa 4.
print("\n1. Base tratada:")
print("Dimensões:", df_tratado.shape)
print("Duplicatas exatas:", int(df_tratado.duplicated().sum()))

# 2. Base analítica original, sem colunas temporárias da EDA.
print("\n2. Base analítica original:")
print("Dimensões:", df_analise.shape)
print("\nSituação dos voos:")
print(df_analise["Situação Voo"].value_counts(dropna=False))

# 3. Base de voos realizados.
print("\n3. Voos realizados:")
print("Total em df_realizados:", len(df_realizados))

print(
    "Registros que não são realizados:",
    int(
        (
            padronizar_categoria(df_realizados["Situação Voo"])
            != "REALIZADO"
        ).sum()
    )
)

# 4. Conferência dos totais após o tratamento.
print("\n4. Conferência dos totais:")

assert len(df_tratado) == 1_005_805, (
    "Verifique a quantidade de registros em df_tratado."
)

assert df_tratado.duplicated().sum() == 0, (
    "Ainda existem duplicatas exatas em df_tratado."
)

assert len(df_analise) == 1_005_805, (
    "Verifique a quantidade de registros em df_analise."
)

assert (
    padronizar_categoria(df_analise["Situação Voo"]) == "REALIZADO"
).sum() == 978_089, (
    "A quantidade de voos realizados está diferente do esperado."
)

assert (
    padronizar_categoria(df_analise["Situação Voo"]) == "CANCELADO"
).sum() == 27_716, (
    "A quantidade de voos cancelados está diferente do esperado."
)

assert len(df_realizados) == 978_089, (
    "Verifique a quantidade de registros em df_realizados."
)

assert padronizar_categoria(
    df_realizados["Situação Voo"]
).eq("REALIZADO").all(), (
    "A base df_realizados contém registros que não são realizados."
)

assert len(df_programados) == 977_607, (
    "A quantidade de registros incluídos na taxa de cancelamento "
    "está diferente do esperado. Verifique as datas e os filtros."
)
assert int(resultado_cancelamento["CANCELADO"].sum()) == 27_716, (
    "A quantidade de cancelamentos incluídos no cálculo não confere."
)

assert (
    int(voos_por_mes.sum()) + voos_sem_partida_real
    == len(df_voos_realizados)
), (
    "O total mensal de voos realizados, somado aos registros "
    "sem Partida Real, não confere com a base."
)

print("\nValidações concluídas: todos os critérios foram atendidos.")

######### ETAPA 5.8 — DASHBOARD INTEGRADO DE OPERAÇÕES AÉREAS #########

# 5.8.1 — VALIDAR OS GRÁFICOS NECESSÁRIOS

graficos_necessarios = [
    "fig_geral",
    "fig_tipo_linha",
    "fig_empresas_dashboard",
    "fig_taxas_dashboard",
    "fig_taxa_cancelamento_rota",
    "fig_heatmap_dashboard",
    "fig_classificacoes",
    "fig_sankey_pontualidade",
    "fig_atrasos_dashboard",
    "fig_mapa_atrasos",
]

for nome_grafico in graficos_necessarios:
    if nome_grafico not in globals():
        raise NameError(
            f"O gráfico '{nome_grafico}' não foi criado. "
            "Verifique se todas as etapas anteriores foram executadas "
            "e se os gráficos possuem os nomes esperados."
        )


######### 5.8.2 — INDICADORES EXECUTIVOS DA VISÃO GERAL #########

# Volume absoluto de voos realizados.
kpi_voos_realizados = int(len(df_voos_realizados))

# Volume absoluto de voos cancelados.
kpi_voos_cancelados = int(mask_cancelado.sum())

# Taxa de cancelamento calculada sobre os voos programados
kpi_total_programado = int(
    resultado_cancelamento["TOTAL PROGRAMADO"].sum()
)

kpi_total_cancelado_avaliavel = int(
    resultado_cancelamento["CANCELADO"].sum()
)

kpi_taxa_cancelamento = (
    kpi_total_cancelado_avaliavel / kpi_total_programado * 100
    if kpi_total_programado > 0
    else 0.0
)

# Indicadores de pontualidade

def calcular_no_horario_ou_antecipado(serie):
    """
    Calcula a proporção de voos antecipados ou pontuais.

    O denominador inclui todos os registros da série,
    inclusive os sem classificação, conforme o critério
    definido para os indicadores executivos.
    """
    if len(serie) == 0:
        return 0.0

    elegiveis = serie.isin(["Antecipado", "Pontual"]).sum()

    return float(elegiveis / len(serie) * 100)


kpi_pontualidade_partida = calcular_no_horario_ou_antecipado(
    df_pontualidade["Classificação Partida"]
)

kpi_pontualidade_chegada = calcular_no_horario_ou_antecipado(
    df_pontualidade["Classificação Chegada"]
)

# HTML dos cartões de indicadores

kpis_visao_geral_html = f"""
<section class="cartoes-kpi" aria-label="Indicadores executivos">

    <article class="cartao-kpi">
        <div class="kpi-rotulo">Voos realizados</div>
        <div class="kpi-valor">{kpi_voos_realizados:,.0f}</div>
        <div class="kpi-detalhe">
            Registros classificados como realizados
        </div>
    </article>

    <article class="cartao-kpi cancelamentos">
        <div class="kpi-rotulo">Voos cancelados</div>
        <div class="kpi-valor">{kpi_voos_cancelados:,.0f}</div>
        <div class="kpi-detalhe">
            Registros classificados como cancelados
        </div>
    </article>

    <article class="cartao-kpi taxa">
        <div class="kpi-rotulo">Taxa de cancelamento</div>
        <div class="kpi-valor">{kpi_taxa_cancelamento:.2f}%</div>
        <div class="kpi-detalhe">
            Sobre {kpi_total_programado:,.0f} voos programados considerados
        </div>
    </article>

    <article class="cartao-kpi pontualidade">
        <div class="kpi-rotulo">Pontualidade operacional</div>

        <div class="kpi-valor">
            {kpi_pontualidade_partida:.1f}%
            <span>partidas</span>
        </div>

        <div class="kpi-valor secundario">
            {kpi_pontualidade_chegada:.1f}%
            <span>chegadas</span>
        </div>

        <div class="kpi-detalhe">
            Voos antecipados ou pontuais, sobre todos os realizados
        </div>
    </article>

</section>
"""

######### 5.8.3 — ORGANIZAR OS GRÁFICOS POR ÁREA ANALÍTICA #########

grupos_dashboard = {

    # VISÃO GERAL

    "visao_geral": {
        "titulo": "📊 Visão geral",
        "subabas": [
            (
                "geral_mensal",
                "Evolução mensal",
                fig_geral,
            ),
            (
                "tipo_linha",
                "Tipo de linha",
                fig_tipo_linha,
            ),
            (
                "empresas",
                "Volume por empresa",
                fig_empresas_dashboard,
            ),
        ],
    },

    # CANCELAMENTOS
    "cancelamentos": {
    "titulo": "⚠️ Cancelamentos",
    "subabas": [
        (
            "taxa_rota",
            "Volume x taxa por rota",
            fig_taxa_cancelamento_rota,
        ),
        (
            "taxas",
            "Taxas por empresa",
            fig_taxas_dashboard,
        ),
        (
            "evolucao_cancelamentos",
            "Evolução mensal",
            fig_heatmap_dashboard,
        ),
    ],
},

    # PONTUALIDADE E ATRASOS

    "atrasos": {
        "titulo": "⏱️ Pontualidade e atrasos",
        "subabas": [
            (
                "classificacoes",
                "Partidas x chegadas",
                fig_classificacoes,
            ),
            (
                "ranking_atrasos",
                "Rankings de atrasos",
                fig_sankey_pontualidade,
            ),
            (
                "fluxo_pontualidade",
                "Fluxo operacional",
                fig_atrasos_dashboard,
            ),
            (
                "mapa_atrasos",
                "Mapa de atrasos por rota",
                fig_mapa_atrasos,
            ),
        ],
    },
}


######### 5.8.4 — CONVERTER OS GRÁFICOS PARA HTML #########

conteudos_graficos = {}

config_graficos = {
    "displayModeBar": True,
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": [
        "lasso2d",
        "select2d",
        "toggleSpikelines",
    ],
}

for grupo_id, grupo in grupos_dashboard.items():

    for grafico_id, titulo_grafico, figura in grupo["subabas"]:

        conteudos_graficos[grafico_id] = pio.to_html(
            figura,
            full_html=False,
            include_plotlyjs=False,
            config=config_graficos,
            div_id=f"grafico_{grafico_id}",
        )


######### 5.8.5 — CRIAR A NAVEGAÇÃO PRINCIPAL #########

botoes_principais = []

for i, (grupo_id, grupo) in enumerate(grupos_dashboard.items()):

    classe = (
        "nav-principal ativo"
        if i == 0
        else "nav-principal"
    )

    botoes_principais.append(
        f'<button class="{classe}" '
        f'data-grupo="{grupo_id}" '
        f'onclick="abrirGrupo(\'{grupo_id}\', this)">'
        f'{grupo["titulo"]}'
        f'</button>'
    )


######### 5.8.6 — CRIAR OS PAINÉIS E AS SUBABAS #########

painels_grupos = []

for i, (grupo_id, grupo) in enumerate(grupos_dashboard.items()):

    classe_grupo = (
        "grupo ativo"
        if i == 0
        else "grupo"
    )

    botoes_subabas = []
    painels_subabas = []

    for j, (grafico_id, titulo_grafico, figura) in enumerate(
        grupo["subabas"]
    ):

        classe_botao = (
            "nav-subaba ativo"
            if j == 0
            else "nav-subaba"
        )

        classe_painel = (
            "painel-grafico ativo"
            if j == 0
            else "painel-grafico"
        )

        # Botão da subaba.
        botoes_subabas.append(
            f'<button class="{classe_botao}" '
            f'data-grafico="{grafico_id}" '
            f'onclick="abrirGrafico('
            f'\'{grupo_id}\', \'{grafico_id}\', this)">'
            f'{titulo_grafico}'
            f'</button>'
        )

        # Painel que contém o gráfico.
        painels_subabas.append(
            f'<section id="{grafico_id}" '
            f'class="{classe_painel}">'
            f'{conteudos_graficos[grafico_id]}'
            f'</section>'
        )

    # Os indicadores executivos aparecem apenas na visão geral.
    indicadores_grupo = (
        kpis_visao_geral_html
        if grupo_id == "visao_geral"
        else ""
    )

    painels_grupos.append(
        f'<div id="grupo_{grupo_id}" class="{classe_grupo}">'
        f'{indicadores_grupo}'
        f'<nav class="abas-secundarias">'
        f'{"".join(botoes_subabas)}'
        f'</nav>'
        f'{"".join(painels_subabas)}'
        f'</div>'
    )

######### 5.8.7 — MONTAR O HTML COMPLETO DO DASHBOARD #########

dashboard_html = f'''<!DOCTYPE html>
<html lang="pt-BR">

<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">

<title>Dashboard de Operações Aéreas — Brasil 2025</title>

<script>
{get_plotlyjs()}
</script>

<style>
* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    padding: 28px;
    background: #F3F6FA;
    color: #24364B;
    font-family: Arial, sans-serif;
}}

.cabecalho {{
    margin: 0 auto 24px;
    max-width: 1500px;
}}

.cabecalho h1 {{
    margin: 0;
    color: #174A7E;
    font-size: clamp(24px, 3vw, 34px);
}}

.cabecalho p {{
    margin: 8px 0 0;
    color: #64748B;
    font-size: 15px;
    line-height: 1.5;
}}

.cartoes-kpi {{
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
    max-width: 1500px;
    margin: 0 auto 22px;
}}

.cartao-kpi {{
    min-width: 0;
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-top: 4px solid #174A7E;
    border-radius: 12px;
    padding: 20px 20px 18px;
    box-shadow: 0 3px 12px rgba(23, 74, 126, 0.05);
}}

.cartao-kpi.cancelamentos {{
    border-top-color: #C0392B;
}}

.cartao-kpi.taxa {{
    border-top-color: #E67E45;
}}

.cartao-kpi.pontualidade {{
    border-top-color: #2F855A;
}}

.kpi-rotulo {{
    color: #64748B;
    font-size: 13px;
    font-weight: 700;
    margin-bottom: 12px;
}}

.kpi-valor {{
    color: #174A7E;
    font-size: clamp(24px, 2.2vw, 32px);
    line-height: 1.2;
    font-weight: 750;
    font-variant-numeric: tabular-nums;
}}

.cartao-kpi.cancelamentos .kpi-valor {{
    color: #C0392B;
}}

.cartao-kpi.taxa .kpi-valor {{
    color: #B95D2A;
}}

.cartao-kpi.pontualidade .kpi-valor {{
    color: #2F855A;
}}

.kpi-valor span {{
    color: #64748B;
    font-size: 12px;
    font-weight: 600;
}}

.kpi-valor.secundario {{
    margin-top: 5px;
    font-size: 22px;
}}

.kpi-detalhe {{
    margin-top: 10px;
    color: #718096;
    font-size: 11px;
    line-height: 1.45;
}}

.navegacao-principal {{
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
    margin: 0 auto 16px;
    max-width: 1500px;
}}

.nav-principal {{
    border: 1px solid #D6E0EC;
    background: #FFFFFF;
    color: #29435E;
    border-radius: 9px;
    padding: 12px 18px;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    transition: background 0.2s, color 0.2s;
}}

.nav-principal:hover {{
    border-color: #174A7E;
}}

.nav-principal.ativo {{
    color: #FFFFFF;
    background: #174A7E;
    border-color: #174A7E;
}}

.grupo {{
    display: none;
    max-width: 1500px;
    margin: 0 auto;
}}

.grupo.ativo {{
    display: block;
}}

.abas-secundarias {{
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    padding: 12px;
    background: #E8EEF5;
    border-radius: 10px 10px 0 0;
}}

.nav-subaba {{
    border: 1px solid transparent;
    background: transparent;
    color: #40566D;
    border-radius: 7px;
    padding: 9px 13px;
    cursor: pointer;
    font-weight: 600;
    font-size: 13px;
    transition: background 0.2s, color 0.2s;
}}

.nav-subaba:hover {{
    color: #174A7E;
}}

.nav-subaba.ativo {{
    background: #FFFFFF;
    color: #174A7E;
    border-color: #D6E0EC;
}}

.painel-grafico {{
    display: none;
    background: #FFFFFF;
    border: 1px solid #E0E7EF;
    border-top: 0;
    border-radius: 0 0 10px 10px;
    padding: 8px;
}}

.painel-grafico.ativo {{
    display: block;
}}

.js-plotly-plot {{
    width: 100%;
}}

.rodape {{
    max-width: 1500px;
    margin: 18px auto 0;
    color: #718096;
    font-size: 12px;
    line-height: 1.6;
}}

@media (max-width: 1000px) {{
    .cartoes-kpi {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }}
}}

@media (max-width: 700px) {{
    body {{
        padding: 12px;
    }}

    .cartoes-kpi {{
        grid-template-columns: 1fr;
        gap: 10px;
    }}

    .cartao-kpi {{
        padding: 16px;
    }}

    .nav-principal {{
        flex: 1 1 auto;
    }}

    .abas-secundarias {{
        gap: 6px;
        padding: 8px;
    }}

    .nav-subaba {{
        padding: 8px 10px;
    }}

    .painel-grafico {{
        padding: 2px;
    }}
}}
</style>
</head>

<body>

<header class="cabecalho">
    <h1>Dashboard de Operações Aéreas — Brasil 2025</h1>
    <p>
        Visão geral da operação, cancelamentos, classificações
        de pontualidade e distribuição dos atrasos por rota aérea.
    </p>
</header>

<nav class="navegacao-principal">
    {"".join(botoes_principais)}
</nav>

<main>
    {"".join(painels_grupos)}
</main>

<p class="rodape">
    Fonte: dados operacionais de voos da ANAC (VRA 2025).
    As taxas de cancelamento utilizam os registros considerados
    válidos para essa análise, conforme os critérios definidos
    nas etapas anteriores. As análises de pontualidade utilizam
    a base de voos realizados. O mapa apresenta as rotas
    selecionadas para a análise de atrasos.
</p>

<script>
function redimensionarGrafico(grupoId, graficoId = null) {{
    let seletor;

    if (graficoId) {{
        seletor = '#' + graficoId + ' .js-plotly-plot';
    }} else {{
        seletor = '#grupo_' + grupoId
            + ' .painel-grafico.ativo .js-plotly-plot';
    }}

    const grafico = document.querySelector(seletor);

    if (grafico && window.Plotly) {{
        setTimeout(function() {{
            Plotly.Plots.resize(grafico);
        }}, 120);
    }}
}}

function abrirGrupo(id, botao) {{
    document.querySelectorAll('.grupo').forEach(function(el) {{
        el.classList.remove('ativo');
    }});

    const grupoSelecionado = document.getElementById('grupo_' + id);

    if (!grupoSelecionado) {{
        return;
    }}

    grupoSelecionado.classList.add('ativo');

    document.querySelectorAll('.nav-principal').forEach(function(el) {{
        el.classList.remove('ativo');
    }});

    botao.classList.add('ativo');

    redimensionarGrafico(id);
}}

function abrirGrafico(grupoId, graficoId, botao) {{
    const grupo = document.getElementById('grupo_' + grupoId);

    if (!grupo) {{
        return;
    }}

    grupo.querySelectorAll('.painel-grafico').forEach(function(el) {{
        el.classList.remove('ativo');
    }});

    grupo.querySelectorAll('.nav-subaba').forEach(function(el) {{
        el.classList.remove('ativo');
    }});

    const painel = document.getElementById(graficoId);

    if (!painel) {{
        return;
    }}

    painel.classList.add('ativo');
    botao.classList.add('ativo');

    redimensionarGrafico(grupoId, graficoId);
}}
</script>

</body>
</html>
'''

######### 5.8.8 — SALVAR O DASHBOARD EM HTML #########

# Pasta do projeto e do portfólio
pasta_projeto = Path(r"C:\Users\User\Desktop\Portifolio")

# Caminho do dashboard
caminho_dashboard = (
    pasta_projeto / "dashboard_operacoes_aereas_2025.html"
)

# Salvar o HTML atualizado
caminho_dashboard.write_text(
    dashboard_html,
    encoding="utf-8",
)

# Confirmar a gravação
if caminho_dashboard.exists():
    print("\n" + "=" * 60)
    print("DASHBOARD INTEGRADO GERADO COM SUCESSO")
    print("=" * 60)
    print(f"\nArquivo salvo em:\n{caminho_dashboard}")
    print(
        f"Tamanho: "
        f"{caminho_dashboard.stat().st_size:,} bytes"
    )
else:
    raise FileNotFoundError(
        f"Não foi possível localizar: {caminho_dashboard}"
    )

######### 5.8.9 — ABRIR O DASHBOARD NO NAVEGADOR #########


if not caminho_dashboard.exists():
    raise FileNotFoundError(
        f"O arquivo HTML não foi encontrado: {caminho_dashboard}"
    )

url_dashboard = caminho_dashboard.as_uri()

# Caminhos comuns de instalação do Google Chrome no Windows
caminhos_chrome = [
    shutil.which("chrome"),
    shutil.which("chrome.exe"),
    os.path.join(
        os.environ.get("PROGRAMFILES", ""),
        "Google",
        "Chrome",
        "Application",
        "chrome.exe",
    ),
    os.path.join(
        os.environ.get("PROGRAMFILES(X86)", ""),
        "Google",
        "Chrome",
        "Application",
        "chrome.exe",
    ),
    os.path.join(
        os.environ.get("LOCALAPPDATA", ""),
        "Google",
        "Chrome",
        "Application",
        "chrome.exe",
    ),
]

chrome_exe = next(
    (
        caminho
        for caminho in caminhos_chrome
        if caminho and os.path.isfile(caminho)
    ),
    None,
)

try:
    if chrome_exe:
        subprocess.Popen([chrome_exe, url_dashboard])
        print("\nChrome iniciado com o dashboard.")

    else:
        print(
            "\nChrome não encontrado nos caminhos habituais."
            "\nAbrindo no navegador padrão do Windows..."
        )

        aberto = webbrowser.open_new_tab(url_dashboard)

        if not aberto and hasattr(os, "startfile"):
            os.startfile(str(caminho_dashboard))

except Exception as erro:
    print(
        "\nNão foi possível abrir o dashboard automaticamente:"
        f"\n{erro}"
    )
    print("\nAbra manualmente o arquivo HTML:")
    print(caminho_dashboard)

print(f"\nArquivo do dashboard: {caminho_dashboard}")