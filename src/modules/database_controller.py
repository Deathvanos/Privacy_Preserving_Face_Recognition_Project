import os
import json
import sqlite3
import numpy as np
from poetry.console.commands import self


class DatabaseController:

    _table_name = "noised_user_components"
    _column_id = "id"
    _column_data = "array"

    def __init__(self, path=r"data\database.db"):
        self.path = os.path.abspath(path)
        print(f"Create db connection at: {self.path}")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Log in the database
        self.conn = sqlite3.connect(self.path)
        self.cursor = self.conn.cursor()
        # Create noised user table
        self.cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS {self._table_name} (
                {self._column_id} INTEGER PRIMARY KEY AUTOINCREMENT,
                {self._column_data} TEXT
            )
        ''')
    def __del__(self):
        # Close database connexions
        self.conn.commit()
        self.conn.close()

    def add_user(self, noised_vectors: np.ndarray):
        json_array = json.dumps(noised_vectors.tolist())
        self.cursor.execute(f"INSERT INTO {self._table_name} ({self._column_data}) VALUES (?)",(json_array,))
        self.conn.commit()
        return self.cursor.lastrowid # Send user id

    def get_user(self, id) -> np.ndarray:
        self.cursor.execute(f"SELECT {self._column_data} FROM {self._table_name} WHERE {self._column_id} = (?)", (id,))
        result = self.cursor.fetchone()
        if result:
            retrieved_json = result[0]
            retrieved_array = np.array(json.loads(retrieved_json))
            return retrieved_array # Send user data
        return None # No user with this id

    def update_user(self, id, new_noised_vectors: np.ndarray):
        json_array = json.dumps(new_noised_vectors.tolist())
        self.cursor.execute(f"UPDATE {self._table_name} SET {self._column_data} = (?) WHERE {self._column_id} = (?)",(json_array, id))
        self.conn.commit()
        return self.cursor.rowcount # Number of affected rows

    def delete_user(self, id):
        self.cursor.execute(f"DELETE FROM {self._table_name} WHERE {self._column_id} = (?)",(id,))
        self.conn.commit()
        return self.cursor.rowcount # Number of affected rows

    def get_user_id_list(self):
        self.cursor.execute(f"SELECT {self._column_id} FROM {self._table_name}")
        result = self.cursor.fetchall()
        return [row[0] for row in result] # Send list of user IDs

    def get_user_vectors(self):
        self.cursor.execute(f"SELECT {self._column_data} FROM {self._table_name}")
        result = self.cursor.fetchall()
        return [row[0] for row in result]

    def get_table(self):
        self.cursor.execute(f"SELECT * FROM {self._table_name}")
        result = self.cursor.fetchall()
        return result


if __name__ == '__main__':

    db = DatabaseController(r"..\..\data\database.db")
    # Add
    data1 = np.random.rand(4, 3)
    print("Original array:\n", data1)
    user_id = db.add_user(data1)
    print("User id is:", user_id)

    # Get 1
    retrieved_array = db.get_user(user_id)
    assert np.array_equal(data1, retrieved_array)
    print("Both arrays match.")

    # Modifie & Get 2
    data2 = np.random.rand(4, 3)
    print("Original array:\n", data2)
    res = db.update_user(user_id, data2)
    print("Number of affected rows:", res)
    retrieved_array = db.get_user(user_id)
    print("Both arrays not match:", data1==retrieved_array)

    # Detele user
    res = db.delete_user(user_id)
    print("Delete user:", user_id)
    print("Number of affected rows:", res)

    user_list = db.get_user_id_list()
    print(user_list)






