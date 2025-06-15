from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import warnings
import pickle
import os
import re
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global değişkenler
model = None
df_clean = None
feature_columns = None

# --- Model Eğitimi ve Kaydetme ---
def train_and_save_model():
    global model, df_clean, feature_columns
    
    print("Model eğitimi başlatılıyor...")
    
    # Sabit Kur Fonksiyonu (INR → TRY)
    def convert_inr_to_try(price_in_inr, rate=0.46):
        return price_in_inr * rate

    # Veri Yükleme
    try:
        df = pd.read_csv("data.csv")
        print(f"Veri yüklendi: {len(df)} satır")
    except FileNotFoundError:
        print("data.csv dosyası bulunamadı! Örnek veri oluşturuluyor...")
        df = create_sample_data()

    # Kullanılacak Sütunlar - CSV'deki gerçek sütunlara göre güncellendi
    columns_to_use = [
        "brand", "name", "price", "spec_rating", "processor", "CPU",
        "Ram", "Ram_type", "ROM", "ROM_type", "GPU", "display_size",
        "resolution_width", "resolution_height", "OS", "warranty"
    ]
    
    # Eksik sütunları kontrol et ve varsayılan değerlerle doldur
    for col in columns_to_use:
        if col not in df.columns:
            if col in ["price", "spec_rating", "display_size", "resolution_width", "resolution_height", "warranty"]:
                df[col] = np.random.randint(1000, 50000) if col == "price" else np.random.uniform(3, 5) if col == "spec_rating" else 15.6
            else:
                df[col] = "Unknown"
    
    df = df[columns_to_use].copy()

    # Fiyatları TL'ye çevir
    df["price"] = df["price"].apply(convert_inr_to_try)

    # Geliştirilmiş sayısal temizlik fonksiyonu
    def extract_number(value):
        if pd.isna(value):
            return 0
        try:
            # String'e çevir ve sayıları çıkar
            str_val = str(value).upper()
            numbers = re.findall(r'\d+', str_val)
            if numbers:
                return int(numbers[0])
            return 0
        except:
            return 0

    # Sayısal sütunları temizle
    df["Ram"] = df["Ram"].apply(extract_number)
    df["ROM"] = df["ROM"].apply(extract_number)
    df["warranty"] = df["warranty"].apply(extract_number)
    
    # 0 değerlerini düzelt
    df.loc[df["Ram"] == 0, "Ram"] = 8  # Varsayılan 8GB
    df.loc[df["ROM"] == 0, "ROM"] = 512  # Varsayılan 512GB
    df.loc[df["warranty"] == 0, "warranty"] = 12  # Varsayılan 12 ay

    # Özellik Mühendisliği
    df["total_pixels"] = df["resolution_width"] * df["resolution_height"]
    df["ppi"] = np.sqrt(df["resolution_width"]**2 + df["resolution_height"]**2) / df["display_size"]
    df["ssd_flag"] = df["ROM_type"].apply(lambda x: 1 if "SSD" in str(x).upper() else 0)

    # Geliştirilmiş RAM hızı skoru
    def ram_type_score(x):
        x = str(x).upper()
        if "DDR5" in x or "LPDDR5" in x:
            return 5
        elif "DDR4" in x or "LPDDR4" in x:
            return 4
        elif "DDR3" in x or "LPDDR3" in x:
            return 3
        elif "UNIFIED" in x:  # Apple için
            return 5
        else:
            return 2

    df["ram_speed_score"] = df["Ram_type"].apply(ram_type_score)

    # Geliştirilmiş marka prestij skoru
    brand_prestige = {
        'APPLE': 5, 'DELL': 4, 'HP': 4, 'LENOVO': 4, 'ASUS': 4,
        'ACER': 3, 'MSI': 4, 'SAMSUNG': 4, 'MICROSOFT': 4,
        'ALIENWARE': 5, 'RAZER': 5, 'THINKPAD': 5, 'GIGABYTE': 4,
        'INFINIX': 2, 'REALME': 2, 'XIAOMI': 3, 'WINGS': 2,
        'ULTIMUS': 1, 'PRIMEBOOK': 1, 'CHUWI': 2, 'TECNO': 2,
        'ZEBRONICS': 2, 'IBALL': 1, 'VAIO': 4, 'HUAWEI': 3,
        'HONOR': 3, 'FUJITSU': 3, 'LG': 4, 'WALKER': 1
    }
    df["brand_prestige"] = df["brand"].str.upper().map(brand_prestige).fillna(2)

    # Geliştirilmiş işlemci performans skoru
    def processor_score(processor):
        processor = str(processor).upper()
        # Intel işlemciler
        if any(x in processor for x in ['I9', 'CORE I9']):
            return 9
        elif any(x in processor for x in ['I7', 'CORE I7']):
            return 7
        elif any(x in processor for x in ['I5', 'CORE I5']):
            return 5
        elif any(x in processor for x in ['I3', 'CORE I3']):
            return 3
        # AMD işlemciler
        elif any(x in processor for x in ['RYZEN 9']):
            return 9
        elif any(x in processor for x in ['RYZEN 7']):
            return 7
        elif any(x in processor for x in ['RYZEN 5']):
            return 5
        elif any(x in processor for x in ['RYZEN 3']):
            return 3
        # Apple işlemciler
        elif any(x in processor for x in ['M2 MAX', 'M1 MAX']):
            return 9
        elif any(x in processor for x in ['M2 PRO', 'M1 PRO']):
            return 8
        elif any(x in processor for x in ['M2', 'M1']):
            return 7
        # Düşük performanslı işlemciler
        elif any(x in processor for x in ['CELERON', 'ATHLON', 'PENTIUM']):
            return 1
        else:
            return 3

    df["processor_performance"] = df["processor"].apply(processor_score)

    # Geliştirilmiş GPU performans skoru
    def gpu_score(gpu):
        gpu = str(gpu).upper()
        # NVIDIA RTX 40 serisi
        if any(x in gpu for x in ['RTX 4090']):
            return 10
        elif any(x in gpu for x in ['RTX 4080']):
            return 9
        elif any(x in gpu for x in ['RTX 4070']):
            return 8
        elif any(x in gpu for x in ['RTX 4060']):
            return 7
        elif any(x in gpu for x in ['RTX 4050']):
            return 6
        # NVIDIA RTX 30 serisi
        elif any(x in gpu for x in ['RTX 3080', 'RTX 3070']):
            return 7
        elif any(x in gpu for x in ['RTX 3060']):
            return 6
        elif any(x in gpu for x in ['RTX 3050']):
            return 5
        # NVIDIA GTX serisi
        elif any(x in gpu for x in ['GTX 1660']):
            return 4
        elif any(x in gpu for x in ['GTX 1650']):
            return 3
        elif any(x in gpu for x in ['GTX 1050']):
            return 2
        # AMD Radeon
        elif any(x in gpu for x in ['RX 6650M', 'RX 6600M']):
            return 5
        elif any(x in gpu for x in ['RX 6500M']):
            return 4
        # Apple GPU
        elif any(x in gpu for x in ['32-CORE GPU', '30-CORE GPU']):
            return 8
        elif any(x in gpu for x in ['16-CORE GPU', '19-CORE GPU']):
            return 7
        elif any(x in gpu for x in ['10-CORE GPU', '8-CORE GPU']):
            return 6
        # Entegre GPU'lar
        elif any(x in gpu for x in ['INTEGRATED', 'INTEL', 'UHD', 'IRIS', 'VEGA', 'RADEON']):
            return 1
        else:
            return 2

    df["gpu_performance"] = df["GPU"].apply(gpu_score)

    # CPU çekirdek skoru
    def cpu_core_score(cpu):
        cpu = str(cpu).upper()
        if 'OCTA CORE' in cpu or '8' in cpu:
            return 4
        elif 'HEXA CORE' in cpu or '6' in cpu:
            return 3
        elif 'QUAD CORE' in cpu or '4' in cpu:
            return 2
        elif 'DUAL CORE' in cpu or '2' in cpu:
            return 1
        else:
            return 2

    df["cpu_core_score"] = df["CPU"].apply(cpu_core_score)

    # Ekran kalitesi skoru
    def screen_score(size, total_pixels):
        if size >= 17 and total_pixels >= 2560*1440:
            return 5
        elif size >= 16 and total_pixels >= 1920*1080:
            return 4
        elif size >= 15 and total_pixels >= 1920*1080:
            return 3
        elif size >= 13 and total_pixels >= 1366*768:
            return 2
        else:
            return 1

    df["screen_quality"] = df.apply(lambda x: screen_score(x["display_size"], x["total_pixels"]), axis=1)

    # İşletim sistemi skoru
    def os_score(os):
        os = str(os).upper()
        if 'MAC' in os:
            return 4
        elif 'WINDOWS 11' in os:
            return 3
        elif 'WINDOWS 10' in os:
            return 2
        elif 'LINUX' in os or 'UBUNTU' in os:
            return 2
        elif 'CHROME' in os:
            return 1
        else:
            return 2

    df["os_score"] = df["OS"].apply(os_score)

    # Performans indeksi
    df["performance_index"] = (
        df["processor_performance"] * 0.25 +
        df["gpu_performance"] * 0.20 +
        df["cpu_core_score"] * 0.15 +
        (df["Ram"] / 32) * 0.15 +  # RAM'i 32GB'a göre normalize et
        (df["ROM"] / 2048) * 0.10 +  # Depolamayı 2TB'a göre normalize et
        df["screen_quality"] * 0.10 +
        df["os_score"] * 0.05
    )

    # Logaritmik dönüşümler
    df["log_ram"] = np.log1p(df["Ram"])
    df["log_rom"] = np.log1p(df["ROM"])
    df["log_pixels"] = np.log1p(df["total_pixels"])

    # RAM/Depolama oranı
    df["ram_rom_ratio"] = df["Ram"] / (df["ROM"] + 1)

    # Kullanılmayan sütunları çıkar
    df.drop(columns=["name"], inplace=True, errors='ignore')

    # Aykırı değer temizleme
    def remove_outliers(df, column, method='iqr'):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    df_clean = remove_outliers(df, 'price', method='iqr')
    print(f"Aykırı değer temizleme sonrası: {len(df_clean)} satır")

    # Özellik & Hedef Ayrımı
    features = df_clean.drop(columns=["price"])
    target = df_clean["price"]

    # Kategorik ve sayısal sütunları ayır
    categorical_cols = features.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = features.select_dtypes(exclude=["object"]).columns.tolist()

    print(f"Kategorik sütunlar: {categorical_cols}")
    print(f"Sayısal sütunlar: {numerical_cols}")

    # Eksik değerleri doldur
    for col in numerical_cols:
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].median())

    for col in categorical_cols:
        if features[col].isnull().any():
            features[col] = features[col].fillna(features[col].mode()[0] if not features[col].mode().empty else 'Unknown')

    # Feature columns'ı sakla
    feature_columns = features.columns.tolist()

    # Pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", RobustScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_cols)
        ]
    )

    # Model eğitimi
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42, stratify=None
    )

    # En iyi modeli bul
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(random_state=42),
        'Ridge': Ridge(random_state=42)
    }

    best_model = None
    best_score = float('inf')
    best_name = ""

    print("Model performansları:")
    for name, model_obj in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", model_obj)
        ])
        
        try:
            cv_scores = cross_val_score(pipeline, X_train, y_train, 
                                       cv=3, scoring='neg_mean_absolute_error')
            mean_cv_score = -cv_scores.mean()
            
            print(f"{name} - CV MAE: {mean_cv_score:.2f}")
            
            if mean_cv_score < best_score:
                best_score = mean_cv_score
                best_model = pipeline
                best_name = name
        except Exception as e:
            print(f"{name} modelinde hata: {e}")

    if best_model is None:
        print("Model eğitimi başarısız!")
        return None, None

    # En iyi modeli eğit
    best_model.fit(X_train, y_train)
    print(f"En iyi model: {best_name}")
    
    # Test performansı
    try:
        test_predictions = best_model.predict(X_test)
        test_mae = mean_absolute_error(y_test, test_predictions)
        test_r2 = r2_score(y_test, test_predictions)
        
        print(f"Test MAE: {test_mae:.2f}")
        print(f"Test R²: {test_r2:.3f}")
    except Exception as e:
        print(f"Test skorları hesaplanırken hata: {e}")
    
    # Modeli kaydet
    try:
        with open('laptop_model.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        print("Model başarıyla kaydedildi!")
    except Exception as e:
        print(f"Model kaydedilirken hata: {e}")
    
    return best_model, df_clean

def create_sample_data():
    """Örnek veri oluşturma fonksiyonu"""
    np.random.seed(42)
    n_samples = 1000
    
    brands = ['HP', 'Dell', 'Lenovo', 'ASUS', 'Acer', 'Apple', 'MSI']
    processors = ['Intel Core i3', 'Intel Core i5', 'Intel Core i7', 'AMD Ryzen 3', 'AMD Ryzen 5', 'AMD Ryzen 7']
    cpus = ['Dual Core', 'Quad Core', 'Hexa Core', 'Octa Core']
    ram_types = ['DDR3', 'DDR4', 'DDR5']
    rom_types = ['HDD', 'SSD', 'eMMC']
    gpus = ['Intel Integrated', 'AMD Radeon', 'NVIDIA GTX 1050', 'NVIDIA RTX 3060', 'NVIDIA RTX 4070']
    os_list = ['Windows 10', 'Windows 11', 'macOS', 'Linux']
    
    data = {
        'brand': np.random.choice(brands, n_samples),
        'name': [f"Laptop {i}" for i in range(n_samples)],
        'price': np.random.randint(15000, 100000, n_samples),
        'spec_rating': np.random.uniform(3.0, 5.0, n_samples),
        'processor': np.random.choice(processors, n_samples),
        'CPU': np.random.choice(cpus, n_samples),
        'Ram': np.random.choice([4, 8, 16, 32], n_samples),
        'Ram_type': np.random.choice(ram_types, n_samples),
        'ROM': np.random.choice([256, 512, 1024, 2048], n_samples),
        'ROM_type': np.random.choice(rom_types, n_samples),
        'GPU': np.random.choice(gpus, n_samples),
        'display_size': np.random.uniform(13.3, 17.3, n_samples),
        'resolution_width': np.random.choice([1366, 1920, 2560], n_samples),
        'resolution_height': np.random.choice([768, 1080, 1440], n_samples),
        'OS': np.random.choice(os_list, n_samples),
        'warranty': np.random.choice([12, 24, 36], n_samples)
    }
    
    return pd.DataFrame(data)

def load_model():
    """Model yükleme fonksiyonu"""
    global model, df_clean, feature_columns
    
    if os.path.exists('laptop_model.pkl'):
        try:
            with open('laptop_model.pkl', 'rb') as f:
                model = pickle.load(f)
            print("Model başarıyla yüklendi!")
            # df_clean'i yeniden oluştur
            _, df_clean = train_and_save_model()
        except Exception as e:
            print(f"Model yüklenirken hata: {e}")
            model, df_clean = train_and_save_model()
    else:
        print("Model bulunamadı, yeni model eğitiliyor...")
        model, df_clean = train_and_save_model()

def predict_laptop_price(user_input):
    """Laptop fiyat tahmini ve öneri fonksiyonu"""
    global model, df_clean, feature_columns
    
    if model is None:
        return 0, []
    
    # Varsayılan değerler
    defaults = {
        "brand": "HP",
        "spec_rating": 4.0,
        "processor": "Intel Core i5",
        "CPU": "Quad Core",
        "Ram": 8,
        "Ram_type": "DDR4",
        "ROM": 512,
        "ROM_type": "SSD",
        "GPU": "Intel Integrated",
        "display_size": 15.6,
        "resolution_width": 1920,
        "resolution_height": 1080,
        "OS": "Windows 11",
        "warranty": 12
    }
    
    # Kullanıcı girdisini varsayılan değerlerle birleştir
    for key, default_value in defaults.items():
        if key not in user_input or user_input[key] == '' or user_input[key] is None:
            user_input[key] = default_value
    
    # Sayısal değerleri dönüştür
    try:
        user_input["spec_rating"] = float(user_input["spec_rating"]) if user_input["spec_rating"] else 4.0
        user_input["Ram"] = int(user_input["Ram"]) if user_input["Ram"] else 8
        user_input["ROM"] = int(user_input["ROM"]) if user_input["ROM"] else 512
        user_input["display_size"] = float(user_input["display_size"]) if user_input["display_size"] else 15.6
        user_input["warranty"] = int(user_input["warranty"]) if user_input["warranty"] else 12
        user_input["resolution_width"] = int(user_input["resolution_width"]) if user_input.get("resolution_width") else 1920
        user_input["resolution_height"] = int(user_input["resolution_height"]) if user_input.get("resolution_height") else 1080
    except Exception as e:
        print(f"Sayısal dönüşüm hatası: {e}")
    
    # Özellik mühendisliği - aynı işlemleri uygula
    user_input["total_pixels"] = user_input["resolution_width"] * user_input["resolution_height"]
    user_input["ppi"] = np.sqrt(user_input["resolution_width"]**2 + user_input["resolution_height"]**2) / user_input["display_size"]
    user_input["ssd_flag"] = 1 if "SSD" in str(user_input["ROM_type"]).upper() else 0
    
    # RAM hızı skoru
    ram_type = str(user_input["Ram_type"]).upper()
    if "DDR5" in ram_type or "LPDDR5" in ram_type:
        user_input["ram_speed_score"] = 5
    elif "DDR4" in ram_type or "LPDDR4" in ram_type:
        user_input["ram_speed_score"] = 4
    elif "DDR3" in ram_type or "LPDDR3" in ram_type:
        user_input["ram_speed_score"] = 3
    elif "UNIFIED" in ram_type:
        user_input["ram_speed_score"] = 5
    else:
        user_input["ram_speed_score"] = 2
    
    # Marka prestij skoru
    brand_prestige = {
        'APPLE': 5, 'DELL': 4, 'HP': 4, 'LENOVO': 4, 'ASUS': 4,
        'ACER': 3, 'MSI': 4, 'SAMSUNG': 4, 'MICROSOFT': 4,
        'ALIENWARE': 5, 'RAZER': 5, 'THINKPAD': 5, 'GIGABYTE': 4
    }
    user_input["brand_prestige"] = brand_prestige.get(str(user_input["brand"]).upper(), 2)
    
    # İşlemci performans skoru - geliştirilmiş
    processor = str(user_input["processor"]).upper()
    if any(x in processor for x in ['I9', 'CORE I9']):
        user_input["processor_performance"] = 9
    elif any(x in processor for x in ['I7', 'CORE I7']):
        user_input["processor_performance"] = 7
    elif any(x in processor for x in ['I5', 'CORE I5']):
        user_input["processor_performance"] = 5
    elif any(x in processor for x in ['I3', 'CORE I3']):
        user_input["processor_performance"] = 3
    elif any(x in processor for x in ['RYZEN 9']):
        user_input["processor_performance"] = 9
    elif any(x in processor for x in ['RYZEN 7']):
        user_input["processor_performance"] = 7
    elif any(x in processor for x in ['RYZEN 5']):
        user_input["processor_performance"] = 5
    elif any(x in processor for x in ['RYZEN 3']):
        user_input["processor_performance"] = 3
    elif any(x in processor for x in ['M2 MAX', 'M1 MAX']):
        user_input["processor_performance"] = 9
    elif any(x in processor for x in ['M2 PRO', 'M1 PRO']):
        user_input["processor_performance"] = 8
    elif any(x in processor for x in ['M2', 'M1']):
        user_input["processor_performance"] = 7
    else:
        user_input["processor_performance"] = 3
    
    # GPU performans skoru - geliştirilmiş
    gpu = str(user_input["GPU"]).upper()
    if 'RTX 4090' in gpu:
        user_input["gpu_performance"] = 10
    elif 'RTX 4080' in gpu:
        user_input["gpu_performance"] = 9
    elif 'RTX 4070' in gpu:
        user_input["gpu_performance"] = 8
    elif 'RTX 4060' in gpu:
        user_input["gpu_performance"] = 7
    elif 'RTX 4050' in gpu:
        user_input["gpu_performance"] = 6
    elif any(x in gpu for x in ['RTX 3080', 'RTX 3070']):
        user_input["gpu_performance"] = 7
    elif 'RTX 3060' in gpu:
        user_input["gpu_performance"] = 6
    elif 'RTX 3050' in gpu:
        user_input["gpu_performance"] = 5
    elif 'GTX 1660' in gpu:
        user_input["gpu_performance"] = 4
    elif 'GTX 1650' in gpu:
        user_input["gpu_performance"] = 3
    elif 'GTX 1050' in gpu:
        user_input["gpu_performance"] = 2
    elif any(x in gpu for x in ['32-CORE GPU', '30-CORE GPU']):
        user_input["gpu_performance"] = 8
    elif any(x in gpu for x in ['16-CORE GPU', '19-CORE GPU']):
        user_input["gpu_performance"] = 7
    elif any(x in gpu for x in ['10-CORE GPU', '8-CORE GPU']):
        user_input["gpu_performance"] = 6
    elif any(x in gpu for x in ['INTEGRATED', 'INTEL', 'UHD', 'IRIS']):
        user_input["gpu_performance"] = 1
    else:
        user_input["gpu_performance"] = 2
    
    # CPU çekirdek skoru
    cpu = str(user_input["CPU"]).upper()
    if 'OCTA CORE' in cpu or '8' in cpu:
        user_input["cpu_core_score"] = 4
    elif 'HEXA CORE' in cpu or '6' in cpu:
        user_input["cpu_core_score"] = 3
    elif 'QUAD CORE' in cpu or '4' in cpu:
        user_input["cpu_core_score"] = 2
    else:
        user_input["cpu_core_score"] = 1
    
    # Ekran kalitesi skoru
    size = user_input["display_size"]
    total_pixels = user_input["total_pixels"]
    if size >= 17 and total_pixels >= 2560*1440:
        user_input["screen_quality"] = 5
    elif size >= 16 and total_pixels >= 1920*1080:
        user_input["screen_quality"] = 4
    elif size >= 15 and total_pixels >= 1920*1080:
        user_input["screen_quality"] = 3
    elif size >= 13 and total_pixels >= 1366*768:
        user_input["screen_quality"] = 2
    else:
        user_input["screen_quality"] = 1
    
    # İşletim sistemi skoru
    os = str(user_input["OS"]).upper()
    if 'MAC' in os:
        user_input["os_score"] = 4
    elif 'WINDOWS 11' in os:
        user_input["os_score"] = 3
    elif 'WINDOWS 10' in os:
        user_input["os_score"] = 2
    else:
        user_input["os_score"] = 2
    
    # Performans indeksi
    user_input["performance_index"] = (
        user_input["processor_performance"] * 0.25 +
        user_input["gpu_performance"] * 0.20 +
        user_input["cpu_core_score"] * 0.15 +
        (user_input["Ram"] / 32) * 0.15 +
        (user_input["ROM"] / 2048) * 0.10 +
        user_input["screen_quality"] * 0.10 +
        user_input["os_score"] * 0.05
    )
    
    # Logaritmik dönüşümler
    user_input["log_ram"] = np.log1p(user_input["Ram"])
    user_input["log_rom"] = np.log1p(user_input["ROM"])
    user_input["log_pixels"] = np.log1p(user_input["total_pixels"])
    
    # RAM/ROM oranı
    user_input["ram_rom_ratio"] = user_input["Ram"] / (user_input["ROM"] + 1)
    
    # DataFrame oluştur
    user_df = pd.DataFrame([user_input])
    
    try:
        # Tahmin yap
        predicted_price = model.predict(user_df)[0]
        
        # Benzer ürünleri bul
        if df_clean is not None and len(df_clean) > 0:
            similar_laptops = df_clean.head(5)
            suggestions = []
            for _, row in similar_laptops.iterrows():
                suggestions.append({
                    "brand": row.get("brand", "Unknown"),
                    "price": row.get("price", 0),
                    "spec_rating": row.get("spec_rating", 0),
                    "CPU": row.get("CPU", "Unknown"),
                    "Ram": row.get("Ram", 0),
                    "ROM": row.get("ROM", 0)
                })
        else:
            suggestions = []
        
        return max(predicted_price, 0), suggestions
        
    except Exception as e:
        print(f"Tahmin hatası: {e}")
        return 0, []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json
        if not user_input:
            return jsonify({
                'success': False,
                'error': 'Veri alınamadı'
            })
        
        if model is None:
            return jsonify({
                'success': False,
                'error': 'Model yüklenmemiş'
            })
        
        predicted_price, suggestions = predict_laptop_price(user_input)
        
        return jsonify({
            'success': True,
            'predicted_price': round(float(predicted_price), 2),
            'suggestions': suggestions
        })
        
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({
            'success': False,
            'error': f'Tahmin hatası: {str(e)}'
        })

# Model yükleme
if __name__ == '__main__':
    load_model()
    app.run(debug=True)