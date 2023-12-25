#!/usr/bin/env python
# coding: utf-8

from sqlalchemy import create_engine, inspect
from sentence_transformers import SentenceTransformer, util
from sqlalchemy.engine.url import URL

import diffusers
import pandas as pd
import safetensors

DATABASE = {
    'drivername': 'postgresql',
    'username': 'postgres',
    'password': 'postgres',
    'host': 'localhost',
    'port': 5432,
    'database': 'postgres',
    'query': {}
}

MODEL_NAME = 'dima-does-code/LaBSE-geonames-15K-MBML-5e-v1'

SELECTED_COUNTRY_CODES = ['RU', 'BY', 'KG', 'KZ', 'AM', 'TR', 'RS']

COUNTRIES_PATH = 'DATA/countryInfo.txt'
CITIES15000_PATH = 'DATA/cities15000.txt'
ADMIN_CODES_PATH = 'DATA/admin1CodesASCII.txt'

class MyGeoClass:

    def __init__(self, model_name=MODEL_NAME, database=DATABASE):
        try:
            # Проверка наличия model_name и database
            if not model_name or not database:
                raise ValueError('Недопустимое имя модели или конфигурация базы данных')

            # Проверка соединения с базой данных
            engine = create_engine(URL(**database))
            if not engine.connect():
                raise ConnectionError('Невозможно подключиться к базе данных')
            self.engine = engine

            self.model_name = model_name
            self.embedder = SentenceTransformer(model_name)

            # Проверка наличия таблицы selected_cities в базе данных
            inspector = inspect(self.engine)
            if 'selected_cities' not in inspector.get_table_names():
                try:
                    print(f"Создание 'selected_cities'...")
                    self.create_selected_cities()
                except Exception as e:
                    print(f"Ошибка при создании 'selected_cities': {e}")
                    raise
            else:
                self.selected_cities = pd.read_sql(
                    'selected_cities', 
                    engine)#, index_col='index')
            
            # Получаем короткое имя модели
            suffix = model_name.split('/')[-1].replace('-', '_').lower()
            # Таблица которую мы хотим проверить
            self.corpus_embeddings_name = f'ce_{suffix}'

            # Проверка наличия таблицы в базе данных
            inspector = inspect(self.engine)
            if self.corpus_embeddings_name not in inspector.get_table_names():
                try:
                    print(f"Создание '{self.corpus_embeddings_name}'...")
                    self.create_corpus_embeddings()
                except Exception as e:
                    print(f"Ошибка при создании '{self.corpus_embeddings_name}': {e}")
                    raise
            else:
                '''
                query = f'SELECT * FROM {self.corpus_embeddings_name}'
                self.corpus_embeddings = pd.read_sql_query(
                    query, con=self.engine
                #).drop('index', axis=1).astype('float32').values
                ).astype('float32').values
                '''
                self.corpus_embeddings = pd.read_sql(
                    self.corpus_embeddings_name,
                    con=self.engine
                ).astype('float32').values

            self.selected_columns = [
                'name_city', 
                'concatenated_codes',
                'name_admin',
                'country'
            ]
            self.new_column_names = [
                'name', 
                'code',
                'region',
                'country', 
                'similarity'
            ]
            
        except Exception as e:
            print(f'Ошибка в __init__: {e}')
            raise


    def my_get_similar(self, query, top_k=1, is_dictionary=False):
        try:
            # Получение векторного представления запроса
            query_embedding = self.embedder.encode(
                query,
                convert_to_tensor=False
            )

            # Поиск семантически близких элементов в корпусе
            hits = util.semantic_search(
                query_embedding, 
                self.corpus_embeddings,
                top_k=top_k
            )
            hits = hits[0]

            # Создание листа для хранения результатов
            result_rows = []

            # Для каждого найденного семантически близкого элемента
            for hit in hits:
                result_row = self.selected_cities.loc[
                    hit['corpus_id'],
                    self.selected_columns
                ].to_frame().T
                result_row['Score'] = hit['score']
                result_rows.append(result_row)

            # Создание итогового DataFrame
            result_df = pd.concat(result_rows, ignore_index=True)
            result_df.columns = self.new_column_names

            # Возвращение результата в виде словаря или DataFrame
            if is_dictionary:
                return result_df.to_dict(orient='records')
            else:
                return result_df

        except Exception as e:
            print(f"Ошибка в 'my_get_similar': {e}")
            raise


    def create_corpus_embeddings(self):
        try:
            # Получение корпуса из выбранных городов
            corpus = self.selected_cities['name_city']

            # Векторное кодирование текстового корпуса городов
            self.corpus_embeddings = self.embedder.encode(
                corpus, 
                convert_to_tensor=False
            )

            # Сохранение векторных эмбеддингов в таблицу
            pd.DataFrame(self.corpus_embeddings).to_sql(
                self.corpus_embeddings_name,
                con=self.engine,
                if_exists='replace',
                index=False
            )

            print(
                "Эмбеддинги корпуса созданы и сохранены в "
                f"'{self.corpus_embeddings_name}'."
            )

        except Exception as e:
            print(f"Ошибка в 'create_corpus_embeddings': {e}")
            raise


    def create_selected_cities(self, selected_country_codes=SELECTED_COUNTRY_CODES):
        try:
            # Проверка наличия таблицы 'countries' в базе данных
            inspector = inspect(self.engine)
            if 'countries' not in inspector.get_table_names():
                try:
                    print(f"Создание 'countries'...")
                    countries = self.read_countries_from_file()
                except Exception as e:
                    print(f"Ошибка при создании 'countries': {e}")
                    raise
            else:
                '''
                query = f'SELECT * FROM countries'
                countries = pd.read_sql_query(query, con=self.engine)
                '''
                countries = pd.read_sql(
                    'countries',
                    con=self.engine
                )

            # Проверка наличия таблицы 'cities15000' в базе данных
            if 'cities15000' not in inspector.get_table_names():
                try:
                    print(f"Создание 'cities15000'...")
                    cities15000 = self.read_cities_from_file()
                except Exception as e:
                    print(f"Ошибка при создании 'cities15000': {e}")
                    raise
            else:
                '''
                query = f'SELECT * FROM cities15000'
                cities15000 = pd.read_sql_query(query, con=self.engine)
                '''
                cities15000 = pd.read_sql(
                    'cities15000',
                    con=self.engine
                )

            # Проверка наличия таблицы 'admin_codes' в базе данных
            if 'admin_codes' not in inspector.get_table_names():
                try:
                    print(f"Создание 'admin_codes'...")
                    admin_codes = self.read_admin_codes_from_file()
                except Exception as e:
                    print(f"Ошибка при создании 'admin_codes': {e}")
                    raise
            else:
                '''
                query = f'SELECT * FROM admin_codes'
                admin_codes = pd.read_sql_query(query, con=self.engine)
                '''
                admin_codes = pd.read_sql(
                    'admin_codes',
                    con=self.engine
                )
            
            admin_codes[
                ['country_code', 'admin1_code']
            ] = admin_codes['concatenated_codes'].str.split('.', expand=True)
            # Объединение данных
            joined_df = pd.merge(
                pd.merge(
                    admin_codes,
                    countries, 
                    on='country_code', 
                    how='left'
                ),
                cities15000, 
                on=['country_code', 'admin1_code'], 
                how='left',
                suffixes=('_admin', '_city') 
            ).dropna()
            
            # Выбор городов по заданным странам
            self.selected_cities = joined_df[
                joined_df[
                    'country_code'
                ].isin(selected_country_codes)].reset_index(drop=True)

            # Сохранение городов по заданным странам в таблицу
            table_name='selected_cities'
            pd.DataFrame(self.selected_cities).to_sql(
                table_name,
                con=self.engine,
                if_exists='replace',
                index=False
            )
            
            print(
                "selected_cities созданы и сохранены в "
                f"'selected_cities'."
            )
        
        except Exception as e:
            print(f"Ошибка в 'create_selected_cities': {e}")
            raise


    def read_countries_from_file(self, countries_path=COUNTRIES_PATH):
        try:
            # Чтение файла с информацией о странах
            countries = pd.read_csv(
                countries_path,
                delimiter='\t',
                header=None,
                comment='#',
                dtype={'geonameid': 'Int64'},
                names=[
                    'country_code',
                    'iso3',
                    'iso-numeric',
                    'fips',
                    'country',
                    'capital',
                    'area(in_sq_km)',
                    'population',
                    'continent',
                    'tld',
                    'currency_code',
                    'currency_name',
                    'phone',
                    'postal_code_format',
                    'postal_code_regex',
                    'languages',
                    'geonameid',
                    'neighbours',
                    'equivalent_fips_code'
                ],
                usecols=[
                    'country_code',
                    'country'
                ]
            ).dropna().drop_duplicates()
            countries.reset_index(drop=True, inplace=True)

            countries.to_sql(
                'countries',
                con=self.engine, 
                if_exists='replace',
                index=False
            )
            return countries

        except Exception as e:
            print(f"Ошибка в 'read_countries_from_file': {e}")
            raise


    def read_cities_from_file(self, cities15000_path=CITIES15000_PATH):
        try:
            # Чтение файла с информацией о городах
            cities15000 = pd.read_csv(
                cities15000_path,
                delimiter='\t',
                header=None,
                index_col=None,
                names=[
                    'geonameid',
                    'name',
                    'asciiname',
                    'alternate_names',
                    'latitude',
                    'longitude',
                    'feature_class',
                    'feature_code',
                    'country_code',
                    'cc2',
                    'admin1_code',
                    'admin2_code',
                    'admin3_code',
                    'admin4_code',
                    'population',
                    'elevation',
                    'dem',
                    'timezone',
                    'modification_date'
                ],
                usecols=[
                    'geonameid',
                    'country_code',
                    'name',
                    'admin1_code',
                    'population'
                ]
            ).dropna().drop_duplicates()
            cities15000.reset_index(drop=True, inplace=True)

            cities15000.to_sql(
                'cities15000',
                con=self.engine,
                if_exists='replace',
                index=False
            )
            return cities15000

        except Exception as e:
            print(f"Ошибка в 'read_cities_from_file': {e}")
            raise


    def read_admin_codes_from_file(self, admin_codes_path=ADMIN_CODES_PATH):
        try:
            # Чтение файла с информацией об административных кодах
            admin_codes = pd.read_csv(
                admin_codes_path,
                delimiter='\t',
                header=None,
                index_col=None,
                names=[
                    'concatenated_codes',
                    'name',
                    'ascii_name',
                    'geonameid'
                ],
                usecols=[
                    'concatenated_codes',
                    'name',
                    # 'ascii_name'
                ]
            ).dropna().drop_duplicates()
            admin_codes.reset_index(drop=True, inplace=True)

            admin_codes.to_sql(
                'admin_codes',
                con=self.engine,
                if_exists='replace',
                index=False
            )
            return admin_codes

        except Exception as e:
            print(f"Ошибка в 'read_admin_codes_from_file': {e}")
            raise


    def __enter__(self):
        return self


    def __exit__(self, exc_type, exc_value, traceback):
        try:
            if hasattr(self, 'engine') and self.engine:
                self.engine.dispose()
        except Exception as e:
            print(f'Ошибка в __exit__: {e}')
