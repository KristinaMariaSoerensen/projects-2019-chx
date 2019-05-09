# Classical Swine Fever - Japan Outbreak 2018/2019 - Data Analysis

See folder **DAP 2.0** for project. The latest dataset was generated the 3rd of May 2019.

This data analysis project analyses and visualizes data on the current outbreak of classical swine fever (CSF) in Japan. 

**Background for project:**

Until September 2018, CSF had been eradicated in Japan since 1992. On September 3rd 2018, cases of CSF on a farm in Gifu Prefecture was once more detected. Since then the Japanese authorities has closely monitored the situation and put in extra precautions to prevent the disease from spreading.

The new updates on outbreaks is only published by the Japanese Ministry for Agriculture, Fisheries and Forestry (MAFF) in Japanese.

The World Organisation for Animal Health: [oie.int](http://www.oie.int/) receives the information from MAFF and publishes a weakly report in English (although due to Japanese holidays it is not always published regularly). The reports are published in the following format: [OIE - Immediate notification (09/09/2018)](http://www.oie.int/wahis_2/public/wahid.php/Reviewreport/Review?reportid=27871)

**Data Structuring**

In the workbook *CSF data generation* I've scraped the html code from all the reports and restructured the info into a dataframe. Futhermore I've collected the coordinates of the outbreaks with GeoLocator so I can make a map. I then save it into a .csv file. For scraping and restructuring method see *CSF data generation*. The running of the code is quite cumbersome, so generating the dataset takes around 15-20 min. 

**Data Analysis**

In this workbook I analyze the data generated in *CSF data generation* by:
- Making a barplot showing outbreaks over time divided on oubreaks among domestic pig (swine) and wild boar
- Making a map plotting all outbreaks
