import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os

st.set_page_config(page_title="Store Sales Forecasting", page_icon="🛒", layout="wide")

BASE_DIR = r"F:\realtime store"

@st.cache_resource
def load_model():
    model        = joblib.load(os.path.join(BASE_DIR, "xgb_model.pkl"))
    preprocessor = joblib.load(os.path.join(BASE_DIR, "preprocessor.pkl"))
    num_col      = joblib.load(os.path.join(BASE_DIR, "num_col.pkl"))
    cat_col      = joblib.load(os.path.join(BASE_DIR, "cat_col.pkl"))
    return model, preprocessor, num_col, cat_col

try:
    model, preprocessor, num_col, cat_col = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)

FAMILIES = ["AUTOMOTIVE","BABY CARE","BEAUTY","BEVERAGES","BOOKS","BREAD/BAKERY","CELEBRATION","CLEANING","DAIRY","DELI","EGGS","FROZEN FOODS","GROCERY I","GROCERY II","HARDWARE","HOME AND KITCHEN I","HOME AND KITCHEN II","HOME APPLIANCES","HOME CARE","LADIESWEAR","LAWN AND GARDEN","LINGERIE","LIQUOR,WINE,BEER","MAGAZINES","MEATS","PERSONAL CARE","PET SUPPLIES","PLAYERS AND ELECTRONICS","POULTRY","PREPARED FOODS","PRODUCE","SCHOOL AND OFFICE SUPPLIES","SEAFOOD"]
CITIES = ["Ambato","Babahoyo","Cayambe","Cuenca","Daule","El Carmen","Esmeraldas","Guaranda","Guayaquil","Ibarra","Latacunga","Libertad","Loja","Machala","Manta","Playas","Puyo","Quevedo","Quito","Riobamba","Salinas","Santo Domingo"]
STATES = ["Azuay","Bolivar","Chimborazo","Cotopaxi","El Oro","Esmeraldas","Guayas","Imbabura","Loja","Los Rios","Manabi","Pastaza","Pichincha","Santa Elena","Santo Domingo de los Tsachilas","Tungurahua"]

st.title("🛒 Store Sales Forecasting")
st.markdown("Du doan doanh thu bang mo hinh **XGBoost** - Corporacion Favorita Ecuador.")
st.divider()

if not model_loaded:
    st.error(f"Khong tai duoc model!\nLoi: {load_error}")
    st.stop()

st.success("Model loaded thanh cong!")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Cua hang")
        store_nbr  = st.slider("Store number", 1, 54, 1)
        store_type = st.selectbox("Loai cua hang", ["A","B","C","D","E"], index=3)
        cluster    = st.slider("Cluster", 1, 17, 8)
        city       = st.selectbox("Thanh pho", CITIES, index=CITIES.index("Quito"))
        state      = st.selectbox("Bang/Tinh", STATES, index=STATES.index("Pichincha"))
    with col2:
        st.subheader("San pham")
        family      = st.selectbox("Nhom hang", FAMILIES)
        onpromotion = st.number_input("So SP khuyen mai", 0, 742, 0)
        isholiday   = st.radio("Ngay le?", [0, 1], format_func=lambda x: "Co" if x else "Khong", horizontal=True)
    with col3:
        st.subheader("Kinh te")
        dcoilwtico       = st.slider("Gia dau (USD)", 26.19, 110.62, 53.19, step=0.5)
        num_transactions = st.number_input("So giao dich", 0, 9000, 1400)
    st.divider()
    submitted = st.form_submit_button("Du doan doanh thu", use_container_width=True, type="primary")

if submitted:
    input_df = pd.DataFrame([{
        "store_nbr": store_nbr, "onpromotion": onpromotion,
        "dcoilwtico": dcoilwtico, "isholiday": isholiday,
        "num_transactions": float(num_transactions), "cluster": cluster,
        "family": family, "city": city, "state": state, "type": store_type
    }])
    try:
        X = preprocessor.transform(input_df[num_col + cat_col]).astype("float32")
        pred = float(np.maximum(model.predict(X)[0], 0))
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Doanh thu du doan", f"{pred:,.2f} don vi")
        c2.metric("Cua hang", f"#{store_nbr} (type {store_type})")
        c3.metric("Nhom hang", family)
        if pred > 5000:
            st.success("Doanh thu cao! Chuan bi hang day du.")
        elif pred == 0:
            st.info("Du doan = 0: Mat hang nay co the khong ban tai cua hang nay ngay do.")
        else:
            st.success("Du doan thanh cong!")
    except Exception as e:
        st.error(f"Loi khi du doan: {e}")

st.divider()
st.caption("Model: XGBoost | Store Sales Time Series Forecasting (Kaggle)")
