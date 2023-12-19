# DL and NLP <a id='geonames'></a>
## Repository of the DL and NLP projects.

<i><b>Project Description</b></i>

<img src="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/geonames.png" width="200" height="200" align="left"/>

This project is focused on the task of associating arbitrary geographical names with standardized GeoNames for internal use by the Career Center. The primary objective is to create a solution capable of efficiently selecting the most appropriate names from the GeoNames database. For instance, it aims to map input names like "Ереван" to their standardized counterparts such as "Yerevan."

The scope of the project extends to countries that are particularly popular for relocation, with a specific focus on Russia and countries like Belarus, Armenia, Kazakhstan, Kyrgyzstan, Turkey, and Serbia. The solution is designed to consider cities with a population of at least 15,000 people, providing scalability options to accommodate the client's server requirements. The output from the solution will include key fields such as geonameid, name, region, country, and cosine similarity, presented in the form of a list of dictionaries representing individual records. This format ensures a structured and accessible output for further analysis and utilization by the Career Center.

| Project Name | Description | Libraries used |
| :---------------------- | :---------------------- | :---------------------- |
| [GeoNames](geonames) | The goal of this project is to develop a solution for mapping geographical names to the standardized names from The GeoNames geographical database for internal use by the Yandex Practicum Career Center.|<i>diffusers, IPython, matplotlib, numpy, pandas, random, safetensors, sentence_transformers, sqlalchemy, warnings</i>