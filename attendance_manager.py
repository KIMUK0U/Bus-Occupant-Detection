import sqlite3
from datetime import datetime

class MinimalAttendanceManager:
    """
    履歴を残さず、ユーザーごとの「現在の状態」と「最終時刻」だけを保持するクラス。
    """

    def __init__(self, db_name="attendance.db"):
        """
        db_name: SQLiteデータベースファイル名
        """
        self.db_name = db_name
        self._create_table()

    def _create_table(self) -> None:
        """
        current_status テーブルが存在しない場合は新規作成する。
        user_id:   ユーザーID (主キー)
        status:    "check_in" / "check_out"
        last_timestamp: その状態になった日時
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS current_status (
            user_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            last_timestamp TEXT NOT NULL
        )
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(create_table_query)
            conn.commit()

    def check_in(self, user_id: str) -> None:
        """
        指定ユーザーを「入室」状態にし、最終時刻を更新する。
        """
        current_time = datetime.now().isoformat()
        insert_query = """
        INSERT INTO current_status (user_id, status, last_timestamp)
        VALUES (?, 'check_in', ?)
        ON CONFLICT(user_id)
        DO UPDATE SET
            status = excluded.status,
            last_timestamp = excluded.last_timestamp
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(insert_query, (user_id, current_time))
            conn.commit()
        print(f"[INFO] {user_id} を check_in 状態にしました: {current_time}")

    def check_out(self, user_id: str) -> None:
        """
        指定ユーザーを「退室」状態にし、最終時刻を更新する。
        """
        current_time = datetime.now().isoformat()
        insert_query = """
        INSERT INTO current_status (user_id, status, last_timestamp)
        VALUES (?, 'check_out', ?)
        ON CONFLICT(user_id)
        DO UPDATE SET
            status = excluded.status,
            last_timestamp = excluded.last_timestamp
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(insert_query, (user_id, current_time))
            conn.commit()
        print(f"[INFO] {user_id} を check_out 状態にしました: {current_time}")

    def get_current_status(self, user_id: str):
        """
        指定したユーザーの (status, last_timestamp) を取得する。
        存在しない場合は None を返す。
        """
        select_query = """
        SELECT status, last_timestamp
        FROM current_status
        WHERE user_id = ?
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(select_query, (user_id,))
            row = cursor.fetchone()
        return row  # (status, last_timestamp) or None

    def delete_user_status(self, user_id: str) -> None:
        """
        指定ユーザーの状態情報を削除する。
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("DELETE FROM current_status WHERE user_id = ?", (user_id,))
            conn.commit()
        print(f"[INFO] ユーザー {user_id} の状態を削除しました。")

    def delete_all_status(self) -> None:
        """
        全ユーザーの状態情報を削除する(テスト用など)。
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute("DELETE FROM current_status")
            conn.commit()
        print("[INFO] 全ユーザーの状態情報を削除しました。")
