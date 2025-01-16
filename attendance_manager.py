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

    def create_user_table(self) -> None:
        """
        登録された顔認証ユーザーの情報を格納するテーブルを作成する。
        user_id: ユーザーID (主キー)
        name: ユーザー名
        """
        create_table_query = """
        CREATE TABLE IF NOT EXISTS registered_users (
            user_id TEXT PRIMARY KEY,
            name TEXT NOT NULL
        )
        """
        with sqlite3.connect(self.db_name) as conn:
            conn.execute(create_table_query)
            conn.commit()

    def register_users(self, user_data: dict) -> None:
        """
        登録ユーザー情報をデータベースに追加する。
        user_data: {user_id: name} の形式の辞書
        """
        insert_query = """
        INSERT INTO registered_users (user_id, name)
        VALUES (?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            name = excluded.name
        """
        with sqlite3.connect(self.db_name) as conn:
            for user_id, name in user_data.items():
                conn.execute(insert_query, (user_id, name))
            conn.commit()
        print(f"[INFO] {len(user_data)} ユーザーを登録しました。")

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

    def update_attendance(self, user_id: str, camera_id: int) -> None:
        """
        指定ユーザーの状態をカメラIDに基づいて更新する。
        camera_id: 0 -> 入室 (check_in), 1 -> 退室 (check_out)
        """
        if camera_id == 0:
            self.check_in(user_id)
        elif camera_id == 1:
            self.check_out(user_id)
        else:
            print(f"[ERROR] 無効なカメラID: {camera_id}")

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

    def show_all_status(self) -> None:
        """
        データベース内の全ユーザーの状態をターミナルに表示する。
        """
        select_query = """
        SELECT user_id, status, last_timestamp
        FROM current_status
        """
        with sqlite3.connect(self.db_name) as conn:
            cursor = conn.execute(select_query)
            rows = cursor.fetchall()

        if rows:
            print("\n[INFO] 現在の全ユーザーの状態:")
            for row in rows:
                print(f"User ID: {row[0]}, Status: {row[1]}, Last Timestamp: {row[2]}")
        else:
            print("\n[INFO] データベースにユーザー状態は存在しません。")


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
