latam_challenge-documentation-test
==============================

This project constitutes a small demonstration on model documentation using data from t

The goal of this project is to develop a supervised machine learning model to solve the classification problem of detecting individuals with annual incomes above 50K USD.

## The Dataset

The problem consists in predicting the probability of delay of the flights that land or take off from the airport of Santiago de Chile (SCL). Each row corresponds to a flight that landed or took off from SCL during 2017. And has been uploaded  into the [data/raw](https://github.com/nelson-io/citi-documentation-test/tree/main/data/raw) The following information is available for each flight:

The Dataset used in this challange is based on US census data from 1994. And has been uploaded  into the [data/raw](https://github.com/alanmatys/latam_challenge/tree/main/data/raw) folder of this repository. It's attributes are the following:

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
