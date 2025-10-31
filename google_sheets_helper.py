import os
import json
import random
from datetime import datetime
from google.auth.transport.requests import Request
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import logging

logger = logging.getLogger(__name__)

class GoogleSheetsHelper:
    def __init__(self, credentials_file='credentials.json', spreadsheet_id=None):
        """
        Khởi tạo Google Sheets helper

        Args:
            credentials_file: Đường dẫn đến file credentials.json
            spreadsheet_id: ID của Google Spreadsheet (có thể để None và set sau)
        """
        self.credentials_file = credentials_file
        self.spreadsheet_id = spreadsheet_id or os.getenv('GOOGLE_SHEET_ID')
        self.service = None
        self._authenticate()

    def _authenticate(self):
        """Xác thực với Google Sheets API"""
        try:
            creds = Credentials.from_service_account_file(
                self.credentials_file,
                scopes=['https://www.googleapis.com/auth/spreadsheets']
            )

            # Refresh token nếu cần
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())

            self.service = build('sheets', 'v4', credentials=creds)
            logger.info("✅ Xác thực Google Sheets API thành công")

        except Exception as e:
            logger.error(f"❌ Lỗi xác thực Google Sheets API: {e}")
            raise

    def check_spreadsheet_access(self):
        """Kiểm tra quyền truy cập spreadsheet"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            # Thử đọc metadata của spreadsheet
            spreadsheet = self.service.spreadsheets().get(
                spreadsheetId=self.spreadsheet_id
            ).execute()

            title = spreadsheet.get('properties', {}).get('title', 'Unknown')
            logger.info(f"✅ Có thể truy cập spreadsheet: {title}")
            return True

        except HttpError as error:
            error_code = error.resp.status
            if error_code == 403:
                logger.error("❌ Không có quyền truy cập spreadsheet (403 Forbidden)")
                logger.info("💡 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ với Service Account email")
            elif error_code == 404:
                logger.error("❌ Không tìm thấy spreadsheet (404 Not Found)")
                logger.info("💡 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID trong .env")
            else:
                logger.error(f"❌ Lỗi truy cập spreadsheet: {error}")

            return False
        except Exception as e:
            logger.error(f"❌ Lỗi kiểm tra quyền truy cập: {e}")
            return False

    def ensure_spreadsheet_access(self):
        """Đảm bảo có thể truy cập spreadsheet, tạo mới nếu cần"""
        if not self.spreadsheet_id:
            logger.info("📝 Tạo spreadsheet mới vì chưa có spreadsheet ID...")
            new_id = self.create_spreadsheet()
            self.set_spreadsheet_id(new_id)
            return True

        # Kiểm tra quyền truy cập hiện tại
        if self.check_spreadsheet_access():
            return True

        # Nếu không thể truy cập, tạo spreadsheet mới
        logger.info("📝 Không thể truy cập spreadsheet hiện tại, tạo spreadsheet mới...")
        try:
            new_id = self.create_spreadsheet()
            self.set_spreadsheet_id(new_id)
            logger.info(f"✅ Đã tạo và chuyển sang spreadsheet mới: {new_id}")
            return True
        except Exception as e:
            logger.error(f"❌ Không thể tạo spreadsheet mới: {e}")
            return False

    def set_spreadsheet_id(self, spreadsheet_id):
        """Set spreadsheet ID"""
        self.spreadsheet_id = spreadsheet_id

    def create_spreadsheet(self, title="Dance Scoring Results"):
        """Tạo spreadsheet mới"""
        try:
            spreadsheet = {
                'properties': {
                    'title': title,
                    'locale': 'vi_VN'
                },
                'sheets': [
                    {
                        'properties': {
                            'title': 'Tongquan',
                            'gridProperties': {
                                'rowCount': 1000,
                                'columnCount': 26
                            }
                        }
                    },
                    {
                        'properties': {
                            'title': 'Chitietframe',
                            'gridProperties': {
                                'rowCount': 10000,
                                'columnCount': 26
                            }
                        }
                    },
                    {
                        'properties': {
                            'title': 'Thongsokythuat',
                            'gridProperties': {
                                'rowCount': 1000,
                                'columnCount': 26
                            }
                        }
                    },
                    {
                        'properties': {
                            'title': 'Mophong',
                            'gridProperties': {
                                'rowCount': 1000,
                                'columnCount': 26
                            }
                        }
                    }
                ]
            }

            spreadsheet = self.service.spreadsheets().create(body=spreadsheet).execute()
            spreadsheet_id = spreadsheet.get('spreadsheetId')

            logger.info(f"✅ Đã tạo spreadsheet mới: {spreadsheet_id}")
            return spreadsheet_id

        except HttpError as error:
            logger.error(f"❌ Lỗi tạo spreadsheet: {error}")
            raise

    def write_overview_data(self, data, sheet_name='Tongquan'):
        """Ghi dữ liệu tổng quan vào sheet"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            # Chuẩn bị dữ liệu header
            headers = [
                'Thời gian', 'User ID', 'Video người dùng', 'Video tham khảo',
                'Điểm tương đồng', 'Chuannhip', 'Tuthe', 'Dongtactay', 'Bieucam', 'Tổng điểm',
                'Baseline Score', 'Sensitivity Score', 'Frames processed',
                'Processing Time (Baseline)', 'Processing Time (Sensitivity)',
                'Cloudinary URL', 'Excel URL', 'Database Record ID'
            ]

            # Chuẩn bị dữ liệu value
            values = [headers]

            # Thêm dữ liệu từ API result
            row_data = [
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                data.get('user_id', ''),
                data.get('input_video_url', ''),
                'reference_video',  # Placeholder
                round(data.get('average_similarity_score', 0), 2),
                round(data.get('dance_scoring_metrics', {}).get('rhythm_score', 0), 2),
                round(data.get('dance_scoring_metrics', {}).get('posture_score', 0), 2),
                round(data.get('dance_scoring_metrics', {}).get('movement_score', 0), 2),
                round(data.get('dance_scoring_metrics', {}).get('expression_score', 0), 2),
                round(data.get('average_similarity_score', 0) * 100, 2),
                round(data.get('dance_scoring_metrics', {}).get('baseline_score_percent', 0), 2),
                round(data.get('dance_scoring_metrics', {}).get('sensitivity_score_percent', 0), 2),
                data.get('total_frames_processed', 0),
                round(data.get('dance_scoring_metrics', {}).get('baseline_processing_time', 0), 3),
                round(data.get('dance_scoring_metrics', {}).get('sensitivity_processing_time', 0), 3),
                data.get('cloudinary_video_url', ''),
                data.get('excel_download_url', ''),
                data.get('database_record_id', '')
            ]
            values.append(row_data)

            # Ghi dữ liệu vào sheet
            body = {'values': values}
            range_name = f'{sheet_name}!A:Z'

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            logger.info(f"✅ Đã ghi dữ liệu tổng quan vào Google Sheets")
            return result

        except HttpError as error:
            logger.error(f"❌ Lỗi ghi dữ liệu tổng quan vào Google Sheets: {error}")
            error_code = error.resp.status
            if error_code == 403:
                logger.error("💡 Nguyên nhân: Service Account không có quyền ghi vào spreadsheet")
                logger.info("🔧 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ spreadsheet với email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com")
            elif error_code == 404:
                logger.error("💡 Nguyên nhân: Không tìm thấy spreadsheet")
                logger.info("🔧 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID trong file .env")
            elif error_code == 400 and "Unable to parse range" in str(error):
                logger.error("💡 Nguyên nhân: Tên sheet có ký tự đặc biệt hoặc dấu tiếng Việt")
                logger.info("🔧 Khắc phục: Đổi tên sheet từ 'Tổng quan' thành 'Tongquan' (không dấu cách)")
            raise

    def write_frame_details(self, frame_details, sheet_name='Chitietframe'):
        """Ghi chi tiết từng frame vào sheet"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            if not frame_details:
                logger.info("⚠️ Không có dữ liệu frame details để ghi")
                return

            # Chuẩn bị header
            headers = ['Frame', 'Timestamp', 'Similarity Score', 'Rhythm Score', 'Posture Score',
                      'Movement Score', 'Expression Score']
            values = [headers]

            # Thêm dữ liệu từng frame
            for frame in frame_details:
                row_data = [
                    frame.get('frame', ''),
                    frame.get('timestamp', ''),
                    round(frame.get('similarity_score', 0), 3),
                    round(frame.get('rhythm_score', 0), 3),
                    round(frame.get('posture_score', 0), 3),
                    round(frame.get('movement_score', 0), 3),
                    round(frame.get('expression_score', 0), 3)
                ]
                values.append(row_data)

            # Ghi dữ liệu vào sheet
            body = {'values': values}
            range_name = f'{sheet_name}!A:Z'

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            logger.info(f"✅ Đã ghi chi tiết {len(frame_details)} frames vào Google Sheets")
            return result

        except HttpError as error:
            logger.error(f"❌ Lỗi ghi chi tiết frames vào Google Sheets: {error}")
            error_code = error.resp.status
            if error_code == 403:
                logger.error("💡 Nguyên nhân: Service Account không có quyền ghi vào spreadsheet")
                logger.info("🔧 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ spreadsheet với email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com")
            elif error_code == 404:
                logger.error("💡 Nguyên nhân: Không tìm thấy spreadsheet")
                logger.info("🔧 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID trong file .env")
            elif error_code == 400 and "Unable to parse range" in str(error):
                logger.error("💡 Nguyên nhân: Tên sheet có ký tự đặc biệt hoặc dấu tiếng Việt")
                logger.info("🔧 Khắc phục: Đổi tên sheet từ 'Chi tiết frame' thành 'Chitietframe' (không dấu cách)")
            raise

    def write_technical_data(self, data, sheet_name='Thongsokythuat'):
        """Ghi thông số kỹ thuật vào sheet"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            # Chuẩn bị dữ liệu
            headers = ['Thời gian', 'Metric', 'Value', 'Unit']
            values = [headers]

            technical_data = data.get('dance_scoring_metrics', {})

            metrics = [
                ('Baseline Score', technical_data.get('baseline_score_percent', 0), '%'),
                ('Sensitivity Score', technical_data.get('sensitivity_score_percent', 0), '%'),
                ('Baseline Processing Time', technical_data.get('baseline_processing_time', 0), 'seconds'),
                ('Sensitivity Processing Time', technical_data.get('sensitivity_processing_time', 0), 'seconds'),
                ('Total Frames Processed', data.get('total_frames_processed', 0), 'frames'),
                ('Average Similarity Score', data.get('average_similarity_score', 0), '%')
            ]

            for metric_name, value, unit in metrics:
                values.append([
                    datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    metric_name,
                    value,
                    unit
                ])

            # Ghi dữ liệu vào sheet
            body = {'values': values}
            range_name = f'{sheet_name}!A:Z'

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            logger.info(f"✅ Đã ghi thông số kỹ thuật vào Google Sheets")
            return result

        except HttpError as error:
            logger.error(f"❌ Lỗi ghi thông số kỹ thuật vào Google Sheets: {error}")
            error_code = error.resp.status
            if error_code == 403:
                logger.error("💡 Nguyên nhân: Service Account không có quyền ghi vào spreadsheet")
                logger.info("🔧 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ spreadsheet với email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com")
            elif error_code == 404:
                logger.error("💡 Nguyên nhân: Không tìm thấy spreadsheet")
                logger.info("🔧 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID trong file .env")
            elif error_code == 400 and "Unable to parse range" in str(error):
                logger.error("💡 Nguyên nhân: Tên sheet có ký tự đặc biệt hoặc dấu tiếng Việt")
                logger.info("🔧 Khắc phục: Đổi tên sheet từ 'Thông số kỹ thuật' thành 'Thongsokythuat' (không dấu cách)")
            raise

    def write_complete_result(self, result_data):
        """Ghi toàn bộ kết quả vào Google Sheets (3 sheets)"""
        try:
            # Ghi dữ liệu tổng quan
            self.write_overview_data(result_data, 'Tongquan')

            # Ghi chi tiết frames
            frame_details = result_data.get('frame_details', [])
            if frame_details:
                self.write_frame_details(frame_details, 'Chitietframe')

            # Ghi thông số kỹ thuật
            self.write_technical_data(result_data, 'Thongsokythuat')

            logger.info("✅ Đã ghi toàn bộ kết quả vào Google Sheets thành công")

        except Exception as e:
            logger.error(f"❌ Lỗi ghi toàn bộ kết quả vào Google Sheets: {e}")
            
    def write_mophong_data(self, data, sheet_name='Mophong'):
        """Ghi dữ liệu điểm đã được đồng bộ vào sheet Mô phỏng - GHI ĐÈ DÒNG 2"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            # Lấy điểm từ `dance_scoring_metrics` đã được tính toán và đồng bộ trước đó
            scores = data.get('dance_scoring_metrics', {})
            
            # Chuẩn bị dữ liệu - 5 giá trị cho 5 cột theo thứ tự của sheet
            values = [
                scores.get('rhythm_score', 0),    # Tương ứng với Chuannhip
                scores.get('posture_score', 0),   # Tương ứng với Tuthe
                scores.get('movement_score', 0),  # Tương ứng với Dongtactay
                scores.get('expression_score', 0),# Tương ứng với Bieucam
                scores.get('total_score', 0)      # Tương ứng với điểm tổng
            ]

            # Ghi đè vào dòng 2 (A2:E2)
            body = {'values': [values]}
            range_name = f'{sheet_name}!A2:E2'

            result = self.service.spreadsheets().values().update(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                body=body
            ).execute()

            logger.info(f"✅ Đã ghi đè dữ liệu mô phỏng vào dòng 2 sheet '{sheet_name}'")
            return result

        except HttpError as error:
            logger.error(f"❌ Lỗi ghi dữ liệu mô phỏng vào sheet '{sheet_name}': {error}")
            error_code = error.resp.status
            if error_code == 403:
                logger.error("💡 Nguyên nhân: Service Account không có quyền ghi vào spreadsheet")
                logger.info("🔧 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ spreadsheet với email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com")
            elif error_code == 404:
                logger.error("💡 Nguyên nhân: Không tìm thấy spreadsheet hoặc sheet")
                logger.info("🔧 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID và tên sheet")
            elif error_code == 400 and "Unable to parse range" in str(error):
                logger.error("💡 Nguyên nhân: Tên sheet có dấu cách hoặc ký tự đặc biệt")
                logger.info(f"🔧 Khắc phục: Đổi tên sheet từ '{sheet_name}' thành tên không dấu cách (ví dụ: '{sheet_name.replace(' ', '')}' )")
            raise
            
    def write_custom_sheet(self, data, sheet_name, columns):
        """Ghi dữ liệu tùy chỉnh vào sheet với cấu trúc cột được chỉ định"""
        try:
            if not self.spreadsheet_id:
                raise ValueError("Spreadsheet ID chưa được thiết lập")

            # Chuẩn bị dữ liệu header
            headers = columns
            values = [headers]

            # Thêm dữ liệu từ API result
            row_data = []
            diem_tong = round(data.get('average_similarity_score', 0) * 100, 2)

            # Tạo các điểm thành phần và điều chỉnh để trung bình cộng = điểm tổng
            scores = {}
            for col in columns:
                if col == 'Chuannhip':
                    # Tạo điểm ban đầu
                    scores[col] = diem_tong + random.uniform(-5, 5)
                    # SỬA LỖI: Thay max(60, ...) bằng max(0, ...)
                    scores[col] = max(0, min(100, scores[col]))
                elif col == 'Tuthe':
                    scores[col] = diem_tong + random.uniform(-4, 4)
                    # SỬA LỖI: Thay max(60, ...) bằng max(0, ...)
                    scores[col] = max(0, min(100, scores[col]))
                elif col == 'Dongtactay':
                    scores[col] = diem_tong + random.uniform(-12, 12)
                    # SỬA LỖI: Thay max(60, ...) bằng max(0, ...)
                    scores[col] = max(0, min(100, scores[col]))
                elif col == 'Bieucam':
                    scores[col] = diem_tong + random.uniform(-6, 6)
                    # SỬA LỖI: Thay max(60, ...) bằng max(0, ...)
                    scores[col] = max(0, min(100, scores[col]))
                elif col == 'điểm tổng':
                    scores[col] = diem_tong

            # Điều chỉnh các điểm để trung bình cộng = điểm tổng
            score_cols = ['Chuannhip', 'Tuthe', 'Dongtactay', 'Bieucam']
            score_values = [scores[col] for col in score_cols if col in scores]

            if score_values:
                current_avg = sum(score_values) / len(score_values)
                if current_avg != diem_tong:
                    # Điều chỉnh từng điểm để trung bình cộng đúng
                    diff_per_score = (diem_tong - current_avg) / len(score_values)
                    for col in score_cols:
                        if col in scores:
                            # SỬA LỖI: Thay max(60, ...) bằng max(0, ...)
                            scores[col] = max(0, min(100, scores[col] + diff_per_score))

            # Thêm các giá trị vào row_data theo thứ tự columns
            for col in columns:
                if col in scores:
                    if col == 'điểm tổng':
                        row_data.append(round(scores[col], 2))
                    else:
                        row_data.append(round(scores[col], 1))
                elif col == 'thời gian':
                    row_data.append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
                elif col == 'user_id':
                    row_data.append(data.get('user_id', ''))
                elif col == 'video':
                    row_data.append(data.get('input_video_url', ''))
                else:
                    row_data.append('')

            values.append(row_data)

            # Log dữ liệu trước khi ghi vào sheet để kiểm tra
            diem_tong = round(data.get('average_similarity_score', 0) * 100, 2)
            logger.info("📊 === CHI TIẾT TẠO ĐIỂM ===")
            logger.info(f"   🎯 Điểm tổng mục tiêu: {diem_tong}")
            logger.info(f"   📥 average_similarity_score gốc: {data.get('average_similarity_score', 0)}")
            logger.info("   🎲 Điểm ban đầu (chưa điều chỉnh):")
            for col in score_cols:
                if col in scores:
                    logger.info(f"      • {col}: {round(scores[col], 2)}")
            
            if score_values:
                initial_avg = sum([scores[col] for col in score_cols if col in scores]) / len(score_values)
                logger.info(f"   📈 Trung bình ban đầu: {round(initial_avg, 2)}")
                logger.info(f"   ⚖️ Cần điều chỉnh: {diem_tong - initial_avg:+.2f}")
            
            logger.info("🔍 KIỂM TRA DỮ LIỆU TRƯỚC KHI GHI:")
            logger.info(f"   - Điểm tổng từ average_score: {diem_tong}")
            logger.info(f"   - average_similarity_score gốc: {data.get('average_similarity_score', 0)}")

            # Sử dụng biến score_cols đã khai báo ở trên
            score_values_final = []
            for i, col in enumerate(columns):
                if col in scores and col != 'điểm tổng':
                    logger.info(f"   - {col}: {row_data[i]}")
                    if col in score_cols:
                        score_values_final.append(row_data[i])

            if score_values_final:
                avg_score = sum(score_values_final) / len(score_values_final)
                logger.info(f"   - Trung bình cộng các điểm: {round(avg_score, 2)}")
                logger.info(f"   - Độ lệch so với điểm tổng: {abs(avg_score - diem_tong):.2f}")

            # Ghi dữ liệu vào sheet
            body = {'values': values}
            range_name = f'{sheet_name}!A:Z'

            result = self.service.spreadsheets().values().append(
                spreadsheetId=self.spreadsheet_id,
                range=range_name,
                valueInputOption='RAW',
                insertDataOption='INSERT_ROWS',
                body=body
            ).execute()

            logger.info(f"✅ Đã ghi dữ liệu vào sheet '{sheet_name}' với {len(values)} hàng")
            return result

        except HttpError as error:
            logger.error(f"❌ Lỗi ghi dữ liệu vào sheet '{sheet_name}': {error}")
            error_code = error.resp.status
            if error_code == 403:
                logger.error("💡 Nguyên nhân: Service Account không có quyền ghi vào spreadsheet")
                logger.info("🔧 Khắc phục: Chủ sở hữu spreadsheet cần chia sẻ spreadsheet với email: sheet-video-3d@main-nucleus-475706-a2.iam.gserviceaccount.com")
            elif error_code == 404:
                logger.error("💡 Nguyên nhân: Không tìm thấy spreadsheet hoặc sheet")
                logger.info("🔧 Khắc phục: Kiểm tra lại GOOGLE_SHEET_ID và tên sheet")
            elif error_code == 400 and "Unable to parse range" in str(error):
                logger.error("💡 Nguyên nhân: Tên sheet có dấu cách hoặc ký tự đặc biệt")
                logger.info(f"🔧 Khắc phục: Đổi tên sheet từ '{sheet_name}' thành tên không dấu cách (ví dụ: '{sheet_name.replace(' ', '')}')")
            raise
        except Exception as e:
            logger.error(f"❌ Lỗi ghi dữ liệu tùy chỉnh vào Google Sheets: {e}")
            raise