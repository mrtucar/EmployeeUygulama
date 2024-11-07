import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder

model = joblib.load("models/best_model.pkl")

st.sidebar.title('Parametreleri seçiniz.')

Education = st.sidebar.selectbox('Eğitim Durumu', ['Bachelors', 'Masters', 'PHD'])
JoiningYear = st.sidebar.slider('İşe Giriş Yılı', 2000, 2020, 2010)
city = st.sidebar.selectbox('Şehir', ['Bangalore', 'Pune', 'New Delhi'])
PaymentTier=st.sidebar.number_input('Ödeme Tiers', min_value=1, max_value=5, value=3, step=1)
Age = st.sidebar.slider('Yaş', 20, 60, 30)
Gender = st.sidebar.radio('Cinsiyet', ['Male','Female'])
EverBenched = st.sidebar.radio('Bench Durumu', ['Yes','No'])
ExperienceInCurrentDomain = st.sidebar.slider('Mevcut Domaindeki Tecrübe', 0, 20, 5)

data = {
    'Education': [Education],
    'JoiningYear': [JoiningYear],
    'City': [city],
    'PaymentTier': [PaymentTier],
    'Age': [Age],
    'Gender': [Gender],
    'EverBenched': [EverBenched],
    'ExperienceInCurrentDomain': [ExperienceInCurrentDomain]
}
df = pd.DataFrame(data)
st.write(df)

sonuc = model.predict(df)
if sonuc[0] == 1:
    st.success('Sonuç: 1 - Çalışan Ayrılacak.')
else:
    st.error('Sonuç: 0 - Çalışan Ayrılmayacak.')