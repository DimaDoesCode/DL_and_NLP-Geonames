[![ru](https://img.shields.io/badge/lang-ru-blue.svg)](https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/README.ru.md)

# DL and NLP <a id='geonames'></a>
## Repository of the DL and NLP projects.

<b>Project Description</b>

<img src="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/geonames.png" width="200" height="200" align="left"/>

This project is focused on the task of associating arbitrary geographical names with standardized GeoNames for internal use by the Career Center. The primary objective is to create a solution capable of efficiently selecting the most appropriate names from the GeoNames database. For instance, it aims to map input names like "Ереван" to their standardized counterparts such as "Yerevan."

The scope of the project extends to countries that are particularly popular for relocation, with a specific focus on Russia and countries like Belarus, Armenia, Kazakhstan, Kyrgyzstan, Turkey, and Serbia. The solution is designed to consider cities with a population of at least 15,000 people, providing scalability options to accommodate the client's server requirements. The output from the solution will include key fields such as geonameid, name, region, country, and cosine similarity, presented in the form of a list of dictionaries representing individual records. This format ensures a structured and accessible output for further analysis and utilization by the Career Center.

**Goal:**
Mapping arbitrary geographical names to standardized GeoNames for internal use by the Career Center.

**Tasks:**
1. Develop a solution to match arbitrary names with GeoNames, e.g., Ереван -> Yerevan.
2. Focus on popular relocation countries: Belarus, Armenia, Kazakhstan, Kyrgyzstan, Turkey, Serbia. Consider cities with a population over 15,000 (scalable on the client's server).
3. Output fields: geonameid, name, region, country, cosine similarity.
4. Output data format: a list of dictionaries, e.g., [{dict_1}, {dict_2}, …, {dict_n}], where each dictionary represents a record with the specified fields.

**Optional Tasks:**
- Allow configuring the number of suggested names (e.g., in method parameters).
- Error correction for typos, e.g., Моченгорск -> Monchegorsk.
- Store GeoNames data in PostgreSQL.
- Store vectorized intermediate data in PostgreSQL.
- Provide methods to configure database connection.
- Include a method for class initialization (initial vectorization of GeoNames).
- Provide methods to add vectors for new geographical names.

**Technology Stack:**
ML Libraries: SQL, Pandas, NLP, Transformers.

**Results:**
1. Notebook with the project solution (project description, research, solution methods).
2. Python script containing a function (class) for integration into the Customer's system.

**Data Description:**
Used GeoNames tables:
- admin1CodesASCII
- alternateNamesV2
- cities15000
- countryInfo

Additional open data if needed.

**Test Dataset:**
File cities15000.txt

---

<a href="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/Geonames_LaBSE.ipynb">To view the Jupyter Notebook code of the research, click on this link.</a><br>
<a href="https://huggingface.co/dima-does-code/LaBSE-geonames-15K-MBML-5e-v1">To view the Model card of the fine tuned LaBSE model used, click on this link.</a><br>
<a href="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/geonames_labse.py">To view the Python Module code of the research, click on this link.</a>

<br>

Example of using the `geonames_labse.py` module:

```python

import geonames_labse

with geonames_labse.MyGeoClass() as my_geo_instance:
    result = my_geo_instance.my_get_similar('Каштана', top_k=5)
```
<br>

| Project Name | Description | Libraries used |
| :---------------------- | :---------------------- | :---------------------- |
| GeoNames | The goal of this project is to develop a solution for mapping geographical names to the standardized names from The GeoNames geographical database for internal use by the Yandex Practicum Career Center.|<i>IPython, matplotlib, numpy, pandas, random, safetensors, seaborn, sentence_transformers, sqlalchemy, warnings</i>