latam_challenge-documentation-test
==============================

The problem consists in predicting the probability of delay of the flights that land or take off from the airport of Santiago de Chile (SCL).

## The Dataset

 Each row corresponds to a flight that landed or took off from SCL during 2017. And has been uploaded  into the [data/raw](https://github.com/nelson-io/citi-documentation-test/tree/main/data/raw) The following information is available for each flight:

* **Fecha-I**: datetime, Scheduled date and time of the flight.
* **Vlo-I**: int, Scheduled flight number.
* **Ori-I**: Programmed origin city code.
* **Des-I**: Programmed destination city code.
* **Emp-I**: Scheduled flight airline code.
* **Fecha-O**: Date and time of flight operation.
* **Vlo-O**: Flight operation number of the flight.
* **Ori-O**: Operation origin city code
* **Des-O**: Operation destination city code.
* **Emp-O**: Airline code of the operated flight.
* **DIA**: int, Day of the month of flight operation.
* **MES**: int, Number of the month of operation of the flight.
* **AÑO**: int, Year of flight operation.
* **DIANOM**: Day of the week of flight operation.
* **TIPOVUELO**: Type of flight, I =International, N =National.
* **OPERA**: Name of the airline that operates.
* **SIGLAORI**: Name city of origin.
* **SIGLADES**: Destination city name.


Project Organization
------------


    
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── interim        <- Intermediate data that has been transformed.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebook with solutions.ipynb and solutions.py (To evaluate changes in commits/push)
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    │
    ├── make_data           <- Scripts to download or generate data
    │        ├── __init__.py
    │        └── input_data.py
    │
    └── docs               <- documentation of the assignment