import mysql.connector
import json
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
import os
from contextlib import contextmanager

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Quản lý kết nối và thao tác với cơ sở dữ liệu MySQL"""

    def __init__(self):
        self.db_config = {
            'host': '103.75.187.176',
            'port': 3308,
            'database': 'nckh_db',
            'user': os.getenv('DB_USERNAME', 'root'),
            'password': os.getenv('DB_PASSWORD', '123456public'),
            'charset': 'utf8mb4',
            'use_pure': True,
            'autocommit': False
        }

        # Không cần connection_params bổ sung cho phiên bản này
        # self.connection_params = {}
        # self.db_config.update(self.connection_params)

    @contextmanager
    def get_connection(self):
        """Context manager để quản lý kết nối database"""
        connection = None
        try:
            connection = mysql.connector.connect(**self.db_config)
            yield connection
        except mysql.connector.Error as e:
            logger.error(f"❌ Database connection error: {e}")
            if connection:
                connection.rollback()
            raise e
        finally:
            if connection and connection.is_connected():
                connection.close()

    def test_connection(self) -> bool:
        """Kiểm tra kết nối database"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                result = cursor.fetchone()
                logger.info("✅ Database connection successful!")
                return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False

    def create_table_if_not_exists(self):
        """Tạo bảng video3d nếu chưa tồn tại"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS video3d (
            id INT AUTO_INCREMENT PRIMARY KEY,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            created_by VARCHAR(255) DEFAULT 'system',
            is_deleted BOOLEAN DEFAULT FALSE,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            updated_by VARCHAR(255) DEFAULT 'system',
            version INT DEFAULT 1,
            json LONGTEXT,
            result_url TEXT,
            title VARCHAR(500),
            video_url TEXT,
            user_id VARCHAR(255),
            process_type ENUM('compare_videos', 'process_video_stream', 'analyze_performance') DEFAULT 'compare_videos',
            status ENUM('processing', 'completed', 'failed') DEFAULT 'processing',
            error_message TEXT,
            processing_time_seconds FLOAT DEFAULT 0,
            type VARCHAR(255),
            INDEX idx_user_id (user_id),
            INDEX idx_process_type (process_type),
            INDEX idx_status (status),
            INDEX idx_created_at (created_at)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                conn.commit()
                logger.info("✅ Table 'video3d' is ready!")
        except mysql.connector.Error as e:
            logger.error(f"❌ Error creating table: {e}")
            raise e

    def save_video_result(self, data: Dict[str, Any], process_type: str, user_id: str = None) -> int:
        """
        Lưu kết quả xử lý video vào database

        Args:
            data: Dữ liệu kết quả từ API
            process_type: Loại xử lý ('compare_videos', 'process_video_stream', 'analyze_performance')
            user_id: ID người dùng (optional)

        Returns:
            ID của bản ghi vừa tạo
        """
        insert_sql = """
        INSERT INTO video3d (
            created_at, created_by, is_deleted, updated_at, updated_by, version,
            json, result_url, title, video_url, user_id, process_type,
            status, error_message, processing_time_seconds, type
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        # Chuẩn bị dữ liệu
        # Special handling for 'process_video_stream' success case as per user request
        if process_type == 'process_video_stream' and 'error' not in data:
            json_payload = {'poses_3d': data.get('poses_3d')}
        else:
            json_payload = data
        
        json_content = json.dumps(json_payload, ensure_ascii=False, default=str)
        result_url = data.get('cloudinary_video_url') or data.get('side_by_side_video_url') or data.get('result_url') or data.get('output_url')
        title = data.get('title') or f"{process_type} - {datetime.now().strftime('%Y%m%d_%H%M%S')}"
        video_url = data.get('video_url') or data.get('input_video_url')
        status = 'completed' if 'error' not in data else 'failed'
        error_message = data.get('error') if 'error' in data else None

        # Normalize process type for enum columns
        process_type_value = (process_type or 'compare_videos').strip()
        if process_type_value not in {'compare_videos', 'process_video_stream', 'analyze_performance'}:
            process_type_value = 'compare_videos'

        if process_type_value == 'process_video_stream':
            record_type = 'PROCESS_VIDEO_STREAM'
        else:
            record_type = 'COMPARE_VIDEO'

        created_by = str(user_id) if user_id is not None else 'system'
        updated_by = created_by
        is_deleted = 0
        version = 1

        # MySQL schema expects integer user_id
        user_id_value = None
        if user_id is not None:
            try:
                user_id_value = int(user_id)
            except (TypeError, ValueError):
                user_id_value = None

        # Tính thời gian xử lý nếu có
        processing_time = 0
        if 'dance_scoring_metrics' in data:
            metrics = data['dance_scoring_metrics']
            processing_time = (
                metrics.get('baseline_processing_time', 0) +
                metrics.get('sensitivity_processing_time', 0)
            )

        logger.info(f"💾 Preparing to save video result to database...")
        logger.info(f"   📋 Process Type: {process_type}")
        logger.info(f"   👤 User ID: {user_id or 'N/A'}")
        logger.info(f"   📝 Title: {title}")
        logger.info(f"   🎬 Video URL: {video_url}")
        logger.info(f"   ✅ Status: {status}")
        logger.info(f"   ⏱️ Processing Time: {processing_time:.3f}s")
        logger.info(f"   📊 JSON Data Size: {len(json_content)} characters")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                logger.info(f"🔗 Executing INSERT query...")

                current_time = datetime.now(timezone.utc)
                cursor.execute(insert_sql, (
                    current_time,
                    created_by,
                    is_deleted,
                    current_time,
                    updated_by,
                    version,
                    json_content,
                    result_url,
                    title,
                    video_url,
                    user_id_value,
                    process_type_value,
                    status,
                    error_message,
                    processing_time,
                    record_type
                ))

                conn.commit()
                record_id = cursor.lastrowid

                logger.info(f"✅ SUCCESS: Video result saved to database!")
                logger.info(f"   🆔 Record ID: {record_id}")
                logger.info(f"   📊 Rows affected: {cursor.rowcount}")

                # Lưu thời gian xử lý nếu có
                # Các trường processing_time/error_message đã được lưu trực tiếp trong INSERT

                return record_id

        except mysql.connector.Error as e:
            logger.error(f"❌ DATABASE ERROR: Failed to save video result!")
            logger.error(f"   🔍 MySQL Error Code: {e.errno}")
            logger.error(f"   📄 Error Message: {e.msg}")
            logger.error(f"   💡 SQL State: {e.sqlstate}")
            raise e

    def update_processing_time(self, record_id: int, processing_time: float):
        """Cập nhật thời gian xử lý cho bản ghi"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE video3d SET processing_time_seconds = %s WHERE id = %s",
                    (processing_time, record_id)
                )
                conn.commit()
        except mysql.connector.Error as e:
            logger.error(f"❌ Error updating processing time: {e}")

    def update_error_message(self, record_id: int, error_message: str):
        """Cập nhật thông báo lỗi cho bản ghi"""
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    "UPDATE video3d SET error_message = %s, status = 'failed' WHERE id = %s",
                    (error_message, record_id)
                )
                conn.commit()
        except mysql.connector.Error as e:
            logger.error(f"❌ Error updating error message: {e}")
    def get_video_results(self, process_type: str = None, user_id: str = None,
                         limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        Lấy danh sách kết quả video từ database

        Args:
            process_type: Loại xử lý để lọc (optional)
            user_id: ID người dùng để lọc (optional)
            limit: Số lượng bản ghi tối đa
            offset: Số bản ghi bỏ qua

        Returns:
            Danh sách kết quả video
        """
        where_conditions = ["is_deleted = FALSE"]
        params = []

        if process_type:
            where_conditions.append("process_type = %s")
            params.append(process_type)

        if user_id:
            where_conditions.append("user_id = %s")
            params.append(user_id)

        where_clause = " AND ".join(where_conditions)

        select_sql = f"""
        SELECT id, created_at, created_by, updated_at, json, result_url,
               title, video_url, user_id, process_type, status, processing_time_seconds, type
        FROM video3d
        WHERE {where_clause}
        ORDER BY created_at DESC
        LIMIT %s OFFSET %s
        """

        params.extend([limit, offset])

        logger.info(f"📖 Querying video results from database...")
        logger.info(f"   🔍 Process Type Filter: {process_type or 'ALL'}")
        logger.info(f"   👤 User ID Filter: {user_id or 'ALL'}")
        logger.info(f"   📊 LIMIT: {limit}, OFFSET: {offset}")
        logger.info(f"   🔗 WHERE Clause: {where_clause}")

        try:
            with self.get_connection() as conn:
                cursor = conn.cursor(dictionary=True)
                logger.info(f"⚡ Executing SELECT query...")

                cursor.execute(select_sql, params)

                results = []
                for row in cursor.fetchall():
                    # Parse JSON data
                    try:
                        json_content = json.loads(row['json']) if row['json'] else {}
                    except json.JSONDecodeError:
                        json_content = {}
                        logger.warning(f"⚠️ Failed to parse JSON data for record {row['id']}")

                    result_item = {
                        'id': row['id'],
                        'created_at': row['created_at'].isoformat() if row['created_at'] else None,
                        'updated_at': row['updated_at'].isoformat() if row['updated_at'] else None,
                        'title': row['title'],
                        'video_url': row['video_url'],
                        'result_url': row['result_url'],
                        'user_id': row['user_id'],
                        'process_type': row['process_type'],
                        'status': row['status'],
                        'processing_time_seconds': row['processing_time_seconds'],
                        'type': row['type'],
                        'data': json_content
                    }
                    results.append(result_item)

                logger.info(f"✅ SUCCESS: Retrieved {len(results)} video results from database")
                logger.info(f"   📋 Records found: {cursor.rowcount}")
                logger.info(f"   📊 Result size: {len(results)} items")

                return results

        except mysql.connector.Error as e:
            logger.error(f"❌ DATABASE ERROR: Failed to retrieve video results!")
            logger.error(f"   🔍 MySQL Error Code: {e.errno}")
            logger.error(f"   📄 Error Message: {e.msg}")
            logger.error(f"   💡 SQL State: {e.sqlstate}")
            raise e

    def get_video_result_by_id(self, record_id: int) -> Optional[Dict[str, Any]]:
        """Lấy thông tin chi tiết của một bản ghi theo ID"""
        try:
            results = self.get_video_results(limit=1, offset=0)
            # Lọc theo ID (do không có where ID trong hàm get_video_results)
            for result in results:
                if result['id'] == record_id:
                    return result
            return None
        except Exception as e:
            logger.error(f"❌ Error retrieving video result by ID: {e}")
            return None

# Global database manager instance
db_manager = DatabaseManager()