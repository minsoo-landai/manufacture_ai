from multiprocessing import Process
from .db_proc import db_proc_worker
from .db_mysql import mysql_connection, ensure_bolt_inspection_table

conn = None

def db_if_mysql_set_conn(in_conn):
    global conn
    conn = in_conn

def db_if_mysql_get_conn():
    global conn
    return conn

def db_if_init():
    conn = mysql_connection(user="cs", password="1234", database="test_db")
    if conn == None:
        return 0
    else:
        db_if_mysql_set_conn(conn)
        ensure_bolt_inspection_table(conn)
        return 1

        
def db_if_run(db_queue, evaluation_queue, config_info, log_file, log_level, log_format):
    try:
        print('db_if : 프로세스 준비')
        process = Process(
            target=db_proc_worker,
            args=(db_queue, evaluation_queue, config_info, log_file, log_level, log_format, db_if_mysql_get_conn())
        )
        print("DB 처리 프로세스 시작")
        return process
            
    except Exception as e:
        print(f"DB 처리 프로세스 시작 실패 : {e}")
        raise
