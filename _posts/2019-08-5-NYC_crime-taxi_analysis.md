---
title: "Analysis of NYC Yellow Taxi trips and NYPD complaints data using Hadoop MapReduce"
data: 2019-08-05
tags: [Big data]
header:
    excerpt: "Big data analytics"
---
<p class="aligncenter">
    <img src="/images/big_data.png" width="300" height="150"/>
</p>

<style>
.aligncenter {
    text-align: center;
}
</style>

# Introduction
These days crimes are increasing at a high rate which propose a   challenge to the law enforcement agencies. A huge amount of data about the different types of crimes is collected and stored annually. The data can be analyzed using big data technologies to find potential solution for the increasing crime rate. As taxis in New York are equipped with GPS sensors a lot of data about the taxi pickup zone, drop off zone, fare amount, etc. is stored.  

Due the large volume of data, the traditional systems find it difficult to process the data. The big data framework can help to discover patterns in data efficiently with great speed. The term big data is referred to data with large volume, velocity and veracity. In big data analytics we look at big data and find patterns, incomprehensible relations and other insights which can be used to prove or disprove assumptions. It is used on a large scale due to its fast development and many frameworks are provided for big data analytics. We deal with big data resources, tools and techniques, big data analytics and its applications.

# Motivation
Safety is crucial in every city. Safety perception might influence people’s behavior and their travel preferences. People might get an idea regarding the safety of their destination and can choose the travelling option according to their convenience. The taxi companies can use this data to improve their service by providing more taxis in the area of greater requirement. Thus, people can check if hiring a taxi is the general trend in that area due to any criminal activity and thus can stay safe by preferring to walk less. Taxi companies can get monetary benefits due to more usage of taxi.  

Understanding crime and its pattern can also be beneficial for people and companies who want to buy a new house or want to start a new business establishment. This also has an impact on walkability score of a region, thus it can have a relationship with taxi pickups and drop-offs. This analytic might be a good stepping stone for prediction of crime counts in a given zone or borough of NY. This might also be useful for organizations like NYPD to curb the crimes in zones with prediction of high crime rate.

## Datasets
1. NYC Yellow Taxi Trips Data 
2. NYPD Complaints Historic Data.

## Implementation
First, collect the taxi and crime datasets and put them in HDFS. To clean the data, use Hadoop Map/Reduce. The cleaned data is then joined using Hive and analytic is performed on Hive. Use Map/Reduce for analytics. Finally, the results are visualized. 

<p class="aligncenter">
    <img src="/images/flowchart.png" width="300" height="150"/>
</p>

## Conclusion
The total number of crime occurrences is positively correlated with the number of pickups by taxi. The reason is people might prefer to use a taxi instead of walking or using public transportation in regions of high crime. It was not possible to fully capture the whole picture due to many different socio-economic factors affecting the taxi pickups in a region. However, the analytic will provide a guideline for other researchers. Addition of different Point of Interest datasets will lead to better understanding of the relationship between crime occurrences and taxi pickups and drop-offs

<a href="https://github.com/asbudhkar/NYC-Crime-and-Taxi">Link to Project:</a>