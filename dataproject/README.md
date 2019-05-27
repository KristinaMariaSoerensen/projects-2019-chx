# Classical Swine Fever - Japan Outbreak 2018/2019 - Data Analysis

This data analysis project analyses and visualizes data on the current outbreak of classical swine fever (CSF) in Japan. 

*The folder **dataproject** contains two notebook files.* 
1. [CSF_data_analysis.ipynb](CSF_data_analysis.ipynb)
2. [CSF_data_generation.ipynb](CSF_data_analysis.ipynb) (OBS - should not be run, see OBS below)
3. [CSF_Japan_data.csv](CSF_Japan_data.csv)

The **results** of the project can be seen from running [CSF_data_analysis.ipynb](CSF_data_analysis.ipynb).

**Dependencies**
Apart from standard packages the project uses the Basemap package from matplotlip.
*from mpl_toolkits.basemap import Basemap*

Can be installed in the terminal using Anaconda as packagemanager by:
' conda install -c conda-forge basemap'

You might also have to install the data-hires files seperately:
' conda install -c conda-forge basemap-data-hires'

**Sources**
[oie.int](http://www.oie.int/)
[maff.go.jp - Info on CSF](http://www.maff.go.jp/j/syouan/douei/csf/)

Inspiration on code and how to solve the model were found at: https://github.com/NumEconCopenhagen and https://github.com/abjer/sds (Course material for Social Data Science)


**OBS:** The [CSF_data_generation.ipynb](CSF_data_analysis.ipynb) is not a part of the hand-in to the exam. Info on this notebook:
*1. It takes around 20 min. to run.
*2. Changes in the html-code of the webpage which is scraped has happened frequently, which unables the data to be generated correctly.
*3. It overwrites the dataset used in the [CSF_data_analysis.ipynb](CSF_data_analysis.ipynb). (Old functioning datasets are included for good measure)
*4. It is included anyway to show the amount of work put behind the dataset.
*5. As of May 26th the latest update from the oie.int was May 17th.

**Data Generation:**
In the workbook [CSF_data_generation.ipynb](CSF_data_analysis.ipynb) I've scraped the html code from all the reports and restructured the info into a dataframe. Futhermore I've collected the coordinates of the outbreaks with GeoLocator so I can make a map. It is then saved into a .csv file. For scraping and restructuring method see [CSF_data_generation.ipynb](CSF_data_analysis.ipynb). The running of the code is quite cumbersome, so generating the dataset takes around 15-20 min. 

**Data Analysis:**
In this workbook I analyze the data generated in [CSF_data_generation.ipynb](CSF_data_analysis.ipynb) by:
- Making a barplot showing outbreaks over time divided on oubreaks among domestic pig (swine) and wild boar
- Making a map plotting all outbreaks