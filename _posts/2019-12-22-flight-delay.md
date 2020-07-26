---
title: "Analyzing relationship between Flight delays and Weather Data for different US airports"
data: 2019-12-22
tags: [Big data]
header:
    excerpt: "Big data analytics"
---
<p class="aligncenter">
    <img src="/images/flight_delays.png" width="300" height="150"/>
</p>

<style>
.aligncenter {
    text-align: center;
}
</style>

Due to the huge growth in air traffic in recent years, it is required to develop effective air transport control systems. Data is collected from ground stations, satellites, sensors on aircraft leading to huge volume of data collected. GPS sensors collect information like distance, time of departure and arrival, etc. Weather stations record information about historical weather as well as predict weather conditions in future. The large volume of data collected from various sources is too big to handle for traditional systems.

As traditional systems find it difficult to process such huge data, big data framework is used to find patterns in data quickly. Big data is considered to have large volume, veracity and velocity. We find patterns, relations within data and other insights in big data analytics. Many frameworks are available for big data analytics. It is useful to prove or disprove assumptions and used on large scale due to fast development.

The flight departure delays were high at JFK from April to August. Similarly, in ORD and DFW showed high number of delays from June to August. While the delays distribution was spread evenly at LAX and SFO, less delays were observed from September to December. Distance wise departure delay distribution was checked monthly which showed a uniform distribution over different ranges of flight distance which leads to a conclusion that flight distance does not show significant effect on the delay.       

As the data distribution was not uniform with non-delayed flight records exceeding four times the number of delayed records, Synthetic Minority Over-sampling Technique was used to generate synthetic samples of minority class and thus make the data distribution uniform for machine learning algorithms. A machine learning model was trained to predict if a flight will be delayed using the weather data and flight information. Weather features – Precipitation, Snow, Wind speed, Max Temperature, Min Temperature and Flight features like arrival delay and flight distance were used to train the model. Logistic Regression, Support vector Machines, Decision Tree and Random Forest classifiers were used.  2017 data was used for training the model while 2018 data was used for testing. The binary classifier was trained for 100 iterations. Best testing accuracy of 91% was obtained for logistic regression model.

<a href="https://github.com/asbudhkar/Spark--Flight-Delay-Analysis-and-Prediction">