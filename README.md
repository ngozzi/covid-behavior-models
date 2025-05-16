# Behavioral-epidemic models for COVID-19
![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)

This repository contains code and data to reproduce the results of the paper "Comparative Evaluation of Behavioral Epidemic Models Using COVID-19 Data" by [Gozzi et al, 2024](https://www.medrxiv.org/content/10.1101/2024.11.08.24316998v1). 

## Data

The data used in the paper is available in the `data` folder. In particular, the `data` folder contains 9 subfolders, each containing the data for a different region considered in the paper (Bogotá, Chicago, Gauteng, Jakarta, London, Madrid, New York, Rio de Janeiro, and Santiago de Chile). Each subfolder contains the following folders/files:
- `contact_matrix`: contains a `contact_matrix.npz` file, which contains the contact matrix for the region. Contact matrices are obtained from [Mistry et al, 2021](https://www.nature.com/articles/s41467-020-20544-y).
- `epi_data`: contains a `epi_data.csv` file, which contains the epidemiological data for the region, with daily new cases and deaths.
- `google-mobility-report`: contains a `google_mobility_data.csv` file, which contains the Google mobility data for the region, from the [Google Mobility Report](https://www.google.com/covid19/mobility/).
- `population-data`: contains a `pop_data_Nk.csv` file, which contains the population in different age groups (0-9,10-19,20-24,25-29,30-39,40-49,50-59,60-69,70-79,80+) for the region.
- `hemisphere`: contains a `hemisphere.csv` file, which contains the hemisphere of the region (needed for seasonal forcing).

The source of the data is the following: 

| Region | Demographic Data Source | Epidemiological Data Source |
|--------|------------------------|----------------------------|
| Bogotá | [Observatorio de Salud de Bogotá, Población de Bogotá](https://saludata.saludcapital.gov.co/osb/indicadores/poblacion-de-bogota-d-c-2005-2035/) | [Gov.co Datos Abiertos, Casos positivos de COVID-19 en Colombia](https://www.datos.gov.co/Salud-y-Protecci-n-Social/Casos-positivos-de-COVID-19-en-Colombia-/gt2j-8ykr/data) |
| Chicago | [Census Reporter, ACS 2022 1-year, Total Population](https://censusreporter.org/data/table/?table=B01001&primary_geo_id=16000US1714000&geo_ids=16000US1714000,05000US17031,31000US16980,04000US17,01000US) | [Chicago Data Portal, Daily Chicago COVID-19 Cases, Deaths, and Hospitalizations - Historical](https://data.cityofchicago.org/Health-Human-Services/Daily-Chicago-COVID-19-Cases-Deaths-and-Hospitaliz/kxzd-kd6a) |
| Gauteng | [Coronavirus COVID-19 (2019-nCoV) Data Repository for South Africa, Provincial projection by sex and age](https://github.com/dsfsi/covid19za/blob/master/data/official_stats/Provincial%20projection%20by%20sex%20and%20age%20(2002-2020)_web.xlsx) | [Coronavirus COVID-19 (2019-nCoV) Data Repository for South Africa](https://github.com/dsfsi/covid19za) |
| Jakarta | [Population by Age Group and Sex in DKI Jakarta Province, 2020](https://jakarta.bps.go.id/id/statistics-table/1/MTQyIzE=/jumlah-penduduk-menurut-kelompok-umur-dan-jenis-kelamin-di-provinsi-dki-jakarta-2015.html) | [Daily Update Data Agregat Covid-19 Jakarta](https://docs.google.com/spreadsheets/d/13oMUqcMijveq00qhSTtQnzJXNuhcdXwDtBRSHQWWLaU/edit?gid=332680197#gid=332680197) |
| London | [Office for National Statistics, Estimates of the population for the UK, England, Wales, Scotland, and Northern Ireland](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/populationestimatesforukenglandandwalesscotlandandnorthernireland) | [Coronavirus (COVID-19) Weekly Update, Greater London Authority (GLA)](https://data.london.gov.uk/dataset/coronavirus--covid-19--cases) |
| Madrid | [Instituto Nacional de Estadistica, Población por comunidades, edad (grupos quinquenales), Españoles/Extranjeros, Sexo y Año](https://www.ine.es/jaxi/Datos.htm?path=/t20/e245/p08/l0/&file=02002.px) | [Ministerio de Sanidad, COVID-19 Deaths](https://raw.githubusercontent.com/datadista/datasets/master/COVID%2019/ccaa_covid19_fallecidos_por_fecha_defuncion_nueva_serie_original.csv) |
| New York | [United States Census Bureau, Age and Sex](https://data.census.gov/table?q=S0101&g=050XX00US36005,36061,36081,36085,36047) | [NYC Health COVID-19 Data](https://www.nyc.gov/site/doh/covid/covid-19-data-totals.page) |
| Rio de Janeiro | [Instituto Brasileiro de Geografia e Estatística, Population Projection](https://www.ibge.gov.br/en/statistics/social/population/18176-population-projection.html) | [Ministério da Saúde, Coronavirus Brazil](https://github.com/henriquemor/covid19-Brazil-timeseries) |
| Santiago de Chile | [Instituto Nacional de Estadisticas, Proyecciones de población](https://www.ine.gob.cl/estadisticas/sociales/demografia-y-vitales/proyecciones-de-poblacion) | [Departamento de Estadísticas e Información de Salud, COVID-19 Open Data](https://deis.minsal.cl/#datosabiertos) |

## Code

The code used to reproduce the results is available in the `models` folder. In particular, the `models` folder contains a file for each of the three behavioral epidemic models considered in the paper:
- `mobility_model_age.py`: contains the code for the Data-Driven Behavioral (DDB) Model.
- `compartment_model_age_deaths.py`: contains the code for the Compartmental Behavioral Feedback (CBF) Model.
- `function_model_age_deaths.py`: contains the code for the Effective Force of Infection Behavioral Feedback (EFB) Model.

Additionally, the file `utils.py` contains the functions used to calibrate the models via Approximate Bayesian Computation (ABC), while the file `constants.py` contains the values of the fixed parameters used in the models.

We provide an example of how to run the models in the [`example.ipynb`](example.ipynb) notebook.
