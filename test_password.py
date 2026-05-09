import psycopg2

passwords = ['postgres', 'password', 'postgre.22', 'admin', 'root', '']

for pwd in passwords:
    try:
        conn = psycopg2.connect(
            host='localhost',
            database='mimic_demo',
            user='postgres',
            password=pwd,
            port=5432
        )
        print(f'✓ SUCCESS! Password is: {pwd}')
        conn.close()
        break
    except Exception as e:
        print(f'✗ Failed with password: {pwd}')