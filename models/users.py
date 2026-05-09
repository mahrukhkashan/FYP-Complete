from werkzeug.security import generate_password_hash, check_password_hash

class User:
    def __init__(self, row):
        self.id = row[0]
        self.username = row[1]
        self.email = row[2]
        self.password_hash = row[3]
        self.role = row[4]

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)
