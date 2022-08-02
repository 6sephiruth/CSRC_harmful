import pymysql


host='143.248.38.246'
user='root'
password='Next_lab2!'
db='hamfulsite'
port=13306

def loadDB():
    connection = pymysql.connect(host=host, user=user, password=password, db=db, port=port, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cur = connection.cursor()
    # sql_check = "SELECT scheme, fld FROM gambleResult where flag=3 and statusCode = 200 LIMIT 1"
    sql_check = "SELECT id, scheme, fld FROM gamble where flag1=1 and id >859"
    sql_check = "SELECT id, scheme, fld FROM gambleResult4 where flag=0"
    print(sql_check)
    cur.execute(sql_check)
    checkResult = cur.fetchall()
    connection.close()
    return checkResult

def updateDB(flag, scheme, fld):
    connection = pymysql.connect(host=host, user=user, password=password, db=db, port=port, charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    cur = connection.cursor()
    sql_update = "UPDATE gambleResult SET flag = %s WHERE scheme='%s' and fld = '%s'"
    print(sql_update % (flag, scheme, fld))
    cur.execute(sql_update % (flag, scheme, fld))
    connection.commit()
    connection.close()

urls = loadDB()
for row in urls:
    url = row['scheme'] + '://' + row['fld'] + '/'

    pid = row['id']
    scheme = row['scheme']
    fld = row['fld']
    print(scheme)
