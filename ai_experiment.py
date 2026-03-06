# Main
database = Database()
database.connect()
errors = database.query("SELECT * FROM errors")
for error in errors:
    print(error)
database.commit()
database.close()