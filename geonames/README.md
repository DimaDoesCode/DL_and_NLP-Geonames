# GeoNames

<b>Customer:</b> Yandex Practicum Career Center.

<b>Project Description:</b>

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

<a href="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/geonames/Geonames_LaBSE.ipynb">To view the Jupyter Notebook code of the research, click on this link.</a><br>
<a href="https://github.com/DimaDoesCode/DL_and_NLP-Geonames/blob/master/geonames/geonames_labse.py">To view the Python Module code of the research, click on this link.</a>

## Libraries used
<i>diffusers, IPython, matplotlib, numpy, pandas, random, safetensors, sentence_transformers, sqlalchemy, warnings</i>