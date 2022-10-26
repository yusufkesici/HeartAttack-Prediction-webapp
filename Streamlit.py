import streamlit as st
import pandas as pd
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression


def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe


def outlier_thresholds(dataframe, col_name, q1=0.20, q3=0.80):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """

    Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

    Parameters
    ------
        dataframe: dataframe
                Değişken isimleri alınmak istenilen dataframe
        cat_th: int, optional
                numerik fakat kategorik olan değişkenler için sınıf eşik değeri
        car_th: int, optinal
                kategorik fakat kardinal değişkenler için sınıf eşik değeri

    Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def HeartAttack_DataPrep(df):
    cat_cols = [col for col in df.columns if df[col].nunique() < 10]
    num_cols = [col for col in df.columns if col not in cat_cols]

    for col in num_cols:
        replace_with_thresholds(df, col)

    # Feature Engineering

    df["Yaş4055"] = np.where((df["Yaş"] > 40) & (df["Yaş"] < 55), 1, 0)

    df["GA_Anjina"] = np.where(((df["Göğüs ağrısı türü"] == 1) | (df["Göğüs ağrısı türü"] == 2))
                               & (df["Egzersize bağlı anjina"] == 0), 1, 0)
    df["GA_Damar"] = np.where(((df["Göğüs ağrısı türü"] == 1) | (df["Göğüs ağrısı türü"] == 2))
                              & (df["Ana damar sayısı"] == 0), 1, 0)
    df["GA_Talasemi"] = np.where(((df["Göğüs ağrısı türü"] == 1) | (df["Göğüs ağrısı türü"] == 2))
                                 & (df["Talasemi"] == 2), 1, 0)

    df["GA_Anjina_other"] = np.where((df["Göğüs ağrısı türü"] == 0) & (df["Egzersize bağlı anjina"] == 1), 1, 0)
    df["Talasemi_Anjina"] = np.where((df["Talasemi"] == 2) & (df["Egzersize bağlı anjina"] == 0), 1, 0)
    df["Damar_Anjina"] = np.where((df["Ana damar sayısı"] == 0) & (df["Egzersize bağlı anjina"] == 0), 1, 0)
    df["Damar_Eğim"] = np.where((df["Ana damar sayısı"] == 0) & (df["Eğim"] == 2), 1, 0)
    df["Talasemi_Eğim"] = np.where((df["Talasemi"] == 2) & (df["Eğim"] == 2), 1, 0)
    df["Talasemi_Damar"] = np.where((df["Talasemi"] == 2) & (df["Ana damar sayısı"] == 0), 1, 0)

    df["kalp/yaş"] = df["Maks kalp hızı"] / df["Yaş"]

    df["Yaş/ST depression"] = df["Yaş"] / (df["ST depression"] + 1)

    df["kalp/ST depression"] = df["Maks kalp hızı"] / (df["ST depression"] + 1)

    df["Kolestoral/ST depression"] = df["Kolestoral"] / (df["ST depression"] + 1)

    df["Koles/kbasınc*kalphız/yaş"] = (df["Kolestoral"] / df["Dinlenme kan basıncı"]) * (
            df["Maks kalp hızı"] / df["Yaş"])

    transform_maxheart = df.groupby(["Cinsiyet", "Egzersize bağlı anjina"])["Maks kalp hızı"].transform("mean")
    df["Kalp_under_mean"] = np.where(df["Maks kalp hızı"] < transform_maxheart, 1, 0)

    ###############################
    # Standart Scaling & Encoding
    ###############################

    cat_cols, num_cols, cat_but_car = grab_col_names(df)

    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])

    ohe_cols = [col for col in cat_cols if df[col].nunique() > 2]
    df = df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

    X = df.drop("Target", axis=1)
    y = df["Target"]

    return X, y


st.set_option("deprecation.showPyplotGlobalUse", False)

st.write("""
# Kalp Krizi Analiz ve Tahmin Uygulaması 
*Accuracy Score: 0.8413, Precision Score: 0.85, Recall Score: 0.8667, F1 Score: 0.8566*
""")

df = pd.read_csv("HeartDataSet/heart.csv")
df.columns = ['Yaş', 'Cinsiyet', 'Göğüs ağrısı türü', 'Dinlenme kan basıncı', 'Kolestoral', 'AKŞ120', 'EKG',
              'Maks kalp hızı', 'Egzersize bağlı anjina', 'ST depression', 'Eğim', 'Ana damar sayısı', 'Talasemi',
              'Target']
# df = pd.read_csv("HeartAttack_Prediction/HeartDataSet/heart.csv")
# df.head()


st.write(df)

# Visulization

chart_select = st.sidebar.selectbox(
    label="Grafik Tipi Seçiniz",
    options=["ScatterPlots", "Histogram", "BoxPlot", "ECDF"]
)

num_cols = [col for col in df.columns if df[col].nunique() > 10]
cat_cols = [col for col in df.columns if col not in num_cols]

if chart_select == "ScatterPlots":
    st.sidebar.subheader("Scatter Plot Ayarları")
    try:
        x_values = st.sidebar.selectbox("X Ekseni", options=num_cols)
        y_values = st.sidebar.selectbox("Y Ekseni", options=num_cols)
        color = st.sidebar.selectbox("Renk", options=cat_cols)
        plot = px.scatter(x=x_values, y=y_values, data_frame=df, color=color)
        st.write(plot)

    except Exception as e:
        print(e)

if chart_select == "Histogram":
    st.sidebar.subheader("Histogram Plot Ayarları")
    try:
        x_values = st.sidebar.selectbox("X Ekseni", options=num_cols)
        plot = px.histogram(df, x=x_values)
        st.write(plot)
    except Exception as e:
        print(e)

if chart_select == "BoxPlot":
    st.sidebar.subheader("Box Plot Ayarları")
    try:
        x_values = st.sidebar.selectbox("X Ekseni", options=num_cols)
        color = st.sidebar.selectbox("Renk", options=cat_cols)
        plot = px.box(df, x=x_values, color=color, notched=True)
        st.write(plot)
    except Exception as e:
        print(e)

if chart_select == "ECDF":
    st.sidebar.subheader("ECDF Plot Ayarları")
    try:
        x_values = st.sidebar.selectbox("X Ekseni", options=num_cols)
        color = st.sidebar.selectbox("Renk", options=cat_cols)
        plot = px.ecdf(df, x=x_values, color=color)
        st.write(plot)
    except Exception as e:
        print(e)

st.sidebar.header("Parametrelerinizi Belirtin")


def user_input_feature():
    Yaş = st.sidebar.slider("Yaş", float(df["Yaş"].min()), float(df["Yaş"].max()), float(df["Yaş"].mean()))
    MaksKP = st.sidebar.slider("Maks kalp hızı", float(df["Maks kalp hızı"].min()), float(df["Maks kalp hızı"].max()),
                               float(df["Maks kalp hızı"].mean()))
    Dinlenmekb = st.sidebar.slider("Dinlenme kan basıncı", float(df["Dinlenme kan basıncı"].min()),
                                   float(df["Dinlenme kan basıncı"].max()), float(df["Dinlenme kan basıncı"].mean()))
    Kolestoral = st.sidebar.slider("Kolestoral", float(df["Kolestoral"].min()), float(df["Kolestoral"].max()),
                                   float(df["Kolestoral"].mean()))
    ST = st.sidebar.slider("ST depression", float(df["ST depression"].min()), float(df["ST depression"].max()), float(df["ST depression"].mean()))
    Cinsiyet = st.sidebar.selectbox("Cinsiyet", (0, 1))
    Göğüs = st.sidebar.selectbox("Göğüs ağrısı türü", (0, 1, 2, 3))
    AKŞ120 = st.sidebar.selectbox("AKŞ120", (0, 1))
    EKG = st.sidebar.selectbox("EKG", (0, 1, 2))
    anjina = st.sidebar.selectbox("Egzersize bağlı anjina", (0, 1))
    Eğim = st.sidebar.selectbox("Eğim", (0, 1, 2))
    damar = st.sidebar.selectbox("Ana damar sayısı", (0, 1, 2, 3, 4))
    Talasemi = st.sidebar.selectbox("Talasemi", (0, 1, 2, 3))
    data = {
        "Yaş": Yaş,
        "Cinsiyet": Cinsiyet,
        "Göğüs ağrısı türü": Göğüs,
        "Dinlenme kan basıncı": Dinlenmekb,
        "Kolestoral": Kolestoral,
        "AKŞ120": AKŞ120,
        "EKG": EKG,
        "Maks kalp hızı": MaksKP,
        "Egzersize bağlı anjina": anjina,
        "ST depression": ST,
        "Eğim": Eğim,
        "Ana damar sayısı": damar,
        "Talasemi": Talasemi
    }

    features = pd.DataFrame(data, index=[0])
    return features


features = user_input_feature()
st.header("Belirtilen Parametreler")
st.write(features)
st.write("---")

features["Target"] = 0

new_df = pd.concat([features, df])

model = pickle.load(open("HeartAttack.pkl", "rb"))

X, y = HeartAttack_DataPrep(new_df)

actual_user = X.head(1)

prediction = model.predict(actual_user)[0]

if prediction == 0:
    prediction = "Kalp Krizi Geçirme İhtimaliniz Düşük"
if prediction == 1:
    prediction = "Kalp Krizi Geçirme İhtimaliniz Yüksek"

st.header("Tahmin Sonuçları")
st.write(prediction)
st.write("---")

