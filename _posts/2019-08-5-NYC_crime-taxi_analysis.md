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

These days crimes are increasing at a high rate which propose a   challenge to the law enforcement agencies. A huge amount of data about the different types of crimes is collected and stored annually. The data can be analyzed using big data technologies to find potential solution for the increasing crime rate. As taxis in New York are equipped with GPS sensors a lot of data about the taxi pickup zone, drop off zone, fare amount, etc. is stored.  

Due the large volume of data, the traditional systems find it difficult to process the data. The big data framework can help to discover patterns in data efficiently with great speed. The term big data is referred to data with large volume, velocity and veracity. In big data analytics we look at big data and find patterns, incomprehensible relations and other insights which can be used to prove or disprove assumptions. It is used on a large scale due to its fast development and many frameworks are provided for big data analytics. We deal with big data resources, tools and techniques, big data analytics and its applications.

First, collect the taxi and crime datasets and put them in HDFS. To clean the data, use Hadoop Map/Reduce. The cleaned data is then joined using Hive and analytic is performed on Hive. Use Map/Reduce for analytics. Finally, the results are visualized. 

<a href="https://github.com/asbudhkar/NYC-Crime-and-Taxi">