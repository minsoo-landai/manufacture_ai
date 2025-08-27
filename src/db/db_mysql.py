import mysql.connector
from mysql.connector import Error
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any, Optional, Iterable

#-----------------------------------
# Mysql 연결
#-----------------------------------
def mysql_connection(
    host: str = "localhost",
    user: str = "root",
    password: str = "your_password",
    database: str = "test_db",
    charset: str = "utf8mb4",
    autocommit: bool = True,
) -> Any: 
    try:
        conn = mysql.connector.connect(
            host=host, user=user, password=password, database=database
        )
        
        conn.set_charset_collation(charset)
        conn.autocommit = autocommit
        return conn
    except Error as e:
        print(f"Mysql 연결 실패:{e}")
        return None

@contextmanager
def cursor(conn):
    cur = None
    try:
        cur = conn.cursor(dictionary=True)
        yield cur
    except Error as e:
        print(f"커서 사용 중 오류:{e}")
    finally:
        try:
            if cur:
                cur.close()
        except Exception:
            pass

# -----------------------------
# 내부 유틸: WHERE/SET 빌더
# -----------------------------
def mysql_build_set_clause(data: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """UPDATE SET 절: dict -> '`col1`=%s, `col2`=%s', [v1, v2]"""
    keys = list(data.keys())
    clause = ", ".join([f"`{k}`=%s" for k in keys])
    params = [data[k] for k in keys]
    return clause, params

def mysql_build_where_clause(
    where: Optional[Dict[str, Any]] = None,
    where_ops: Optional[Dict[str, str]] = None,
    extra: Optional[str] = None,
) -> Tuple[str, List[Any]]:
    """
    WHERE 절 생성기
    - where: {"id": 1, "status": "A"}
    - where_ops: {"age": ">=", "score": ">" }  # 기본 '='
    - extra: "updated_at >= NOW()" 같은 추가 SQL 조각(바인딩 없는 형태 권장)
    """
    if not where and not extra:
        return "", []
    parts, params = [], []
    if where:
        for k, v in where.items():
            op = "="
            if where_ops and k in where_ops:
                op = where_ops[k].strip().upper()
            parts.append(f"`{k}` {op} %s")
            params.append(v)
    clause = " WHERE " + " AND ".join(parts) if parts else ""
    if extra:
        clause += (" AND " if clause else " WHERE ") + extra
    return clause, params

# -----------------------------
# INSERT
# -----------------------------
def mysql_insert_row(conn, table: str, data: Dict[str, Any]) -> Optional[int]:
    """
    단건 INSERT. 성공 시 lastrowid, 실패 시 None.
    """
    try:
        cols = ", ".join([f"`{k}`" for k in data.keys()])
        placeholders = ", ".join(["%s"] * len(data))
        sql = f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})"
        with cursor(conn) as cur:
            if cur is None:
                return None
            cur.execute(sql, list(data.values()))
        try:
            conn.commit()
        except Exception:
            pass
        return cur.lastrowid if cur else None
    except Error as e:
        print(f"insert_row 오류({table}): {e}")
        return None

def mysql_insert_many(conn, table: str, rows: Iterable[Dict[str, Any]]) -> Optional[int]:
    """
    다건 INSERT (모든 row가 같은 키 셋을 가정). 성공 시 삽입된 행 수, 실패 시 None.
    """
    try:
        rows = list(rows)
        if not rows:
            return 0
        keys = list(rows[0].keys())
        cols = ", ".join([f"`{k}`" for k in keys])
        placeholders = ", ".join(["%s"] * len(keys))
        sql = f"INSERT INTO `{table}` ({cols}) VALUES ({placeholders})"
        params = [tuple(row[k] for k in keys) for row in rows]
        with cursor(conn) as cur:
            if cur is None:
                return None
            cur.executemany(sql, params)
            affected = cur.rowcount
        try:
            conn.commit()
        except Exception:
            pass
        return affected
    except Error as e:
        print(f"insert_many 오류({table}): {e}")
        return None

# -----------------------------
# UPSERT (있으면 업데이트, 없으면 삽입)
# -----------------------------
def mysql_upsert_row(conn, table: str, data: Dict[str, Any], key_cols: List[str]) -> Optional[int]:
    """
    MySQL ON DUPLICATE KEY UPDATE 활용. 성공 시 영향 행 수(rowcount), 실패 시 None.
    key_cols: UNIQUE/PK 컬럼명들
    """
    try:
        cols = list(data.keys())
        insert_cols = ", ".join([f"`{c}`" for c in cols])
        placeholders = ", ".join(["%s"] * len(cols))
        update_set = ", ".join([f"`{c}`=VALUES(`{c}`)" for c in cols if c not in key_cols])
        sql = f"""
        INSERT INTO `{table}` ({insert_cols})
        VALUES ({placeholders})
        ON DUPLICATE KEY UPDATE {update_set}
        """
        with cursor(conn) as cur:
            if cur is None:
                return None
            cur.execute(sql, [data[c] for c in cols])
            affected = cur.rowcount
        try:
            conn.commit()
        except Exception:
            pass
        return affected
    except Error as e:
        print(f"upsert_row 오류({table}): {e}")
        return None

# -----------------------------
# UPDATE
# -----------------------------
def mysql_update_rows(
    conn,
    table: str,
    data: Dict[str, Any],
    where: Optional[Dict[str, Any]] = None,
    where_ops: Optional[Dict[str, str]] = None,
    extra_where: Optional[str] = None,
    require_where: bool = True,  # 안전장치: WHERE 없으면 기본 차단
) -> Optional[int]:
    """
    dict 기반 UPDATE. 성공 시 영향 행 수(rowcount), 실패 시 None.
    - require_where=True 이고 where/extra_where 둘 다 없으면 실행하지 않음.
    """
    try:
        if not data:
            print("update_rows: data가 비어 있습니다.")
            return 0

        # 안전장치: WHERE 없는 전체 업데이트 방지
        if require_where and not where and not extra_where:
            print("update_rows 차단: WHERE 없이 전체 업데이트 시도")
            return None

        set_clause, set_params = mysql_build_set_clause(data)
        where_clause, where_params = mysql_build_where_clause(where, where_ops, extra_where)
        sql = f"UPDATE `{table}` SET {set_clause}{where_clause}"

        with cursor(conn) as cur:
            if cur is None:
                return None
            cur.execute(sql, set_params + where_params)
            affected = cur.rowcount
        try:
            conn.commit()
        except Exception:
            pass
        return affected
    except Error as e:
        print(f"update_rows 오류({table}): {e}")
        return None

# -----------------------------
# DELETE
# -----------------------------
def mysql_delete_rows(
    conn,
    table: str,
    where: Optional[Dict[str, Any]] = None,
    where_ops: Optional[Dict[str, str]] = None,
    extra_where: Optional[str] = None,
    require_where: bool = True,  # 안전장치: WHERE 없으면 기본 차단
) -> Optional[int]:
    """
    DELETE. 성공 시 영향 행 수(rowcount), 실패 시 None.
    - require_where=True 이고 where/extra_where 둘 다 없으면 실행하지 않음.
    """
    try:
        # 안전장치: WHERE 없는 전체 삭제 방지
        if require_where and not where and not extra_where:
            print("delete_rows 차단: WHERE 없이 전체 삭제 시도")
            return None

        where_clause, where_params = mysql_build_where_clause(where, where_ops, extra_where)
        sql = f"DELETE FROM `{table}`{where_clause}"

        with cursor(conn) as cur:
            if cur is None:
                return None
            cur.execute(sql, where_params)
            affected = cur.rowcount
        try:
            conn.commit()
        except Exception:
            pass
        return affected
    except Error as e:
        print(f"delete_rows 오류({table}): {e}")
        return None

# -----------------------------
# SELECT
# -----------------------------
def mysql_select_rows(
    conn,
    table: str,
    columns: Iterable[str] = ("*",),
    where: Optional[Dict[str, Any]] = None,
    where_ops: Optional[Dict[str, str]] = None,
    extra_where: Optional[str] = None,
    order_by: Optional[str] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    SELECT. 실패 시 빈 리스트 반환.
    """
    try:
        cols = ", ".join(columns)
        where_clause, where_params = mysql_build_where_clause(where, where_ops, extra_where)
        sql = f"SELECT {cols} FROM `{table}`{where_clause}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit is not None:
            sql += " LIMIT %s"
            where_params.append(limit)
            if offset is not None:
                sql += " OFFSET %s"
                where_params.append(offset)

        with cursor(conn) as cur:
            if cur is None:
                return []
            cur.execute(sql, where_params)
            return cur.fetchall()
    except Error as e:
        print(f"select_rows 오류({table}): {e}")
        return []

# -----------------------------
# DB 스키마
# -----------------------------
def mysql_execute(conn, sql: str, params=None):
    from mysql.connector import Error
    try:
        with cursor(conn) as cur:
            if cur is None:
                return False
            cur.execute(sql, params or [])
        try:
            conn.commit()
        except Exception:
            pass
        return True
    except Error as e:
        print(f"SQL 실행 실패: {e}")
        return False

def mysql_is_alive(conn) -> bool:
    """
    연결 여부 체크
    """
    try:
        conn.ping(reconnect=True, attempts=1, delay=0)
        return True
    except Exception:
        return False

def ensure_index(conn, table: str, index_name: str, column_expr: str) -> bool:
    """
    인덱스 안전하게 생성하는 유틸 (버전 에러 방지)
    - column_expr 예: "`capture_datetime`" 또는 "`raw_audio_path`(191)"
    """
    try:
        sql_check = """
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = DATABASE()
          AND table_name = %s
          AND index_name = %s
        LIMIT 1
        """
        with cursor(conn) as cur:
            cur.execute(sql_check, [table, index_name])
            exists = cur.fetchone() is not None
        if exists:
            return True
        # 없으면 생성
        return mysql_execute(conn,
            f"ALTER TABLE `{table}` ADD INDEX `{index_name}` ({column_expr})"
        )
    except Exception as e:
        print(f"ensure_index 실패: {e}")
        return False

def ensure_bolt_inspection_table(conn):
    ddl = """
    CREATE TABLE IF NOT EXISTS `bolt_inspection` (
      `id`                 INT AUTO_INCREMENT PRIMARY KEY,
      `capture_datetime`   DATETIME                 NULL,
      `raw_vibration_data` MEDIUMBLOB              NULL,
      `raw_audio_data`     LONGBLOB                NULL,
      `raw_audio_path`     VARCHAR(500)            NULL,
      `dtw_score`          FLOAT                   NULL,
      `ae_loss`            FLOAT                   NULL,
      `final_score`        FLOAT                   NULL,
      `sample_rate`        INT                     NULL,
      `bolt_status`        ENUM('양품','불량품')   NULL,
      `created_at`         TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
    """
    ok = mysql_execute(conn, ddl)

    # MySQL 8+ 안전 패턴: IF NOT EXISTS 지원
    ok &= ensure_index(conn, "bolt_inspection", "idx_capture_datetime", "`capture_datetime`")
    ok &= ensure_index(conn, "bolt_inspection", "idx_bolt_status", "`bolt_status`")

    # 경로 인덱스(prefix 191)
    ok &= ensure_index(conn, "bolt_inspection", "idx_path_prefix", "`raw_audio_path`(191)")

    return ok
