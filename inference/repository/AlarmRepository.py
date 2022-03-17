import psycopg2
from database.config import config

from datetime import datetime

from entity.AlarmEntity import AlarmEntity


class AlarmRepository(object):
    def __init__(self) -> None:
        self.entity: AlarmEntity = None

    def insert(self) -> None:
        conn = None
        statement = "INSERT INTO alarm(city, country, status, threshold, created_on, updated_on, audio_path, classe, model_type) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        try:
            params = config()
            print("Connecting to the PostgreSQL database...")
            conn = psycopg2.connect(**params)
            cursor = conn.cursor()
            cursor.execute(statement, (self.entity.city,
                                       self.entity.country,
                                       self.entity.status,
                                       self.entity.threshold,
                                       self.entity.created_on,
                                       self.entity.updated_on,
                                       self.entity.audio_path,
                                       self.entity.classe,
                                       self.entity.model_type))
            conn.commit()
            cursor.close()
        except (Exception, psycopg2.DatabaseError) as e:
            print(e)
        finally:
            if conn is not None:
                conn.close()
                print("Database connection closed")

    def setEntity(self, entity) -> None:
        self.entity = entity