#!/usr/bin/env python3
"""
Script test tích hợp Google Sheets API
Chạy script này để kiểm tra xem Google Sheets integration có hoạt động đúng không
"""

import os
import sys
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_google_sheets_integration():
    """Test Google Sheets integration"""
    try:
        logger.info("🧪 Bắt đầu test Google Sheets integration...")

        # Import Google Sheets helper
        from google_sheets_helper import GoogleSheetsHelper

        # Khởi tạo helper
        logger.info("🔧 Khởi tạo GoogleSheetsHelper...")
        sheets_helper = GoogleSheetsHelper()

        # Test tạo spreadsheet mới
        logger.info("📝 Tạo spreadsheet test...")
        test_spreadsheet_id = sheets_helper.create_spreadsheet("Test Dance Scoring Results")
        logger.info(f"✅ Đã tạo spreadsheet với ID: {test_spreadsheet_id}")

        # Set spreadsheet ID để sử dụng
        sheets_helper.set_spreadsheet_id(test_spreadsheet_id)

        # Test dữ liệu mẫu
        test_data = {
            'user_id': 'test_user_123',
            'input_video_url': 'test_video.mp4',
            'average_similarity_score': 85.5,
            'dance_scoring_metrics': {
                'rhythm_score': 82.0,
                'posture_score': 88.0,
                'movement_score': 84.0,
                'expression_score': 87.0,
                'total_score': 85.25,
                'baseline_score_percent': 92.0,
                'sensitivity_score_percent': 88.0,
                'baseline_processing_time': 2.5,
                'sensitivity_processing_time': 1.8
            },
            'total_frames_processed': 150,
            'cloudinary_video_url': 'https://res.cloudinary.com/test/video.mp4',
            'excel_download_url': '/res/output/test_dance_scoring.xlsx',
            'database_record_id': 'test_record_123',
            'frame_details': [
                {
                    'frame': 1,
                    'timestamp': '00:00:01',
                    'similarity_score': 85.0,
                    'rhythm_score': 82.0,
                    'posture_score': 88.0,
                    'movement_score': 84.0,
                    'expression_score': 87.0
                },
                {
                    'frame': 2,
                    'timestamp': '00:00:02',
                    'similarity_score': 86.0,
                    'rhythm_score': 83.0,
                    'posture_score': 89.0,
                    'movement_score': 85.0,
                    'expression_score': 88.0
                }
            ]
        }

        # Test ghi dữ liệu vào các sheet
        logger.info("📊 Ghi dữ liệu tổng quan...")
        sheets_helper.write_overview_data(test_data, 'Tổng quan')

        logger.info("📋 Ghi chi tiết frames...")
        sheets_helper.write_frame_details(test_data['frame_details'], 'Chi tiết frame')

        logger.info("⚙️ Ghi thông số kỹ thuật...")
        sheets_helper.write_technical_data(test_data, 'Thông số kỹ thuật')

        logger.info("✅ TẤT CẢ TESTS THÀNH CÔNG!")
        logger.info(f"🎉 Spreadsheet test đã sẵn sàng: https://docs.google.com/spreadsheets/d/{test_spreadsheet_id}/edit")

        return True

    except Exception as e:
        logger.error(f"❌ Test thất bại: {e}")
        logger.error(f"🔍 Chi tiết lỗi: {sys.exc_info()}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Bắt đầu kiểm tra tích hợp Google Sheets...")

    # Kiểm tra file credentials
    if not os.path.exists('credentials.json'):
        logger.error("❌ Không tìm thấy file credentials.json")
        logger.info("ℹ️ Vui lòng đặt file credentials.json vào thư mục gốc của dự án")
        return False

    # Kiểm tra biến môi trường
    if not os.getenv('GOOGLE_SHEET_ID'):
        logger.warning("⚠️ Chưa thiết lập GOOGLE_SHEET_ID trong .env")
        logger.info("ℹ️ Sẽ tạo spreadsheet test mới")

    # Chạy test
    success = test_google_sheets_integration()

    if success:
        logger.info("🎉 Tích hợp Google Sheets hoạt động hoàn hảo!")
        logger.info("💡 Bạn có thể bắt đầu sử dụng API với Google Sheets integration")
    else:
        logger.error("❌ Tích hợp Google Sheets có vấn đề")
        logger.info("💡 Hãy kiểm tra lại cấu hình và thử lại")

    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
