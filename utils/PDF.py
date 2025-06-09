import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from scipy.stats import chi2_contingency, pearsonr
from utils.ChartStorage import ChartStorage
from utils.Data import Data
from utils.DataAnalyzer import DataAnalyzer

from fpdf import FPDF
import base64
import tempfile
from datetime import datetime
import os
import unicodedata # Thêm thư viện này để xử lý tiếng Việt không dấu

class PDF(FPDF):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Không cần đăng ký font tiếng Việt nữa, dùng font Arial mặc định của FPDF
        self.set_font("Arial", size=12)
        st.info("Báo cáo PDF sẽ được tạo với các ký tự không dấu (ASCII).")
        
    def header(self):
        pass

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8) # Dùng Arial cho footer
        self.cell(0, 10, 'Trang %s' % self.page_no(), 0, 0, 'C')
