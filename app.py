import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# Configuración de la página de Streamlit
st.set_page_config(page_title="EDA Catálogo de Parámetros", layout="wide")
st.title("Análisis Exploratorio de Datos (EDA) - Catálogo de Parámetros")
sns.set_theme(style="whitegrid")
# 1. CARGAR EL ARCHIVO DESDE LA INTERFAZ WEB
uploaded_file = st.file_uploader("Sube tu archivo cat_parametros.csv", type=["csv"])
if uploaded_file is not None:
    # El archivo tiene una fila de título antes de los encabezados.
    df_original = pd.read_csv(
        uploaded_file,
        encoding="latin1",
        skiprows=1
    )
    st.markdown("---")
    st.header("1. Inspección Inicial")
    st.write(f"**Dimensiones originales:** {df_original.shape[0]} filas y {df_original.shape[1]} columnas")
    st.subheader("Primeras cinco filas:")
    st.dataframe(df_original.head())
    with st.expander("Ver información general del DataFrame (.info())"):
        import io
        buffer = io.StringIO()
        df_original.info(buf=buffer)
        st.text(buffer.getvalue())
    st.write("**Columnas:**", df_original.columns.tolist())
    # 2. LIMPIEZA BASICA
    df = df_original.copy()
    df.columns = df.columns.str.strip()
    for columna in df.select_dtypes(include="object").columns:
        df[columna] = df[columna].str.strip()
    df["id_parameter"] = pd.to_numeric(df["id_parameter"], errors="coerce")
    st.markdown("---")
    st.header("2. Valores Faltantes y Limpieza")
    faltantes = pd.DataFrame({
        "Cantidad": df.isna().sum(),
        "Porcentaje": (df.isna().mean() * 100).round(2)
    })
    st.dataframe(faltantes)
    st.write(f"**Filas duplicadas exactas:** {df.duplicated().sum()}")
    df = df.drop_duplicates().copy()
    st.subheader("Claves de parámetro repetidas:")
    repetidas = df[df["cve_param"].duplicated(keep=False)].sort_values("cve_param")
    if not repetidas.empty:
        st.dataframe(repetidas)
    else:
        st.success("No hay claves de parámetro repetidas.")
    st.write(f"Dimensiones antes: {df_original.shape} | Dimensiones después: {df.shape}")
    # 3. ESTADISTICA DESCRIPTIVA
    st.markdown("---")
    st.header("3. Estadística Descriptiva")
    st.subheader("Resumen de variables numéricas:")
    st.dataframe(df.describe().T)
    st.subheader("Resumen de variables categóricas:")
    st.dataframe(df.describe(include="object").T)
    col1, col2, col3 = st.columns(3)
    col1.metric("Parámetros diferentes (cve_param)", df["cve_param"].nunique())
    col2.metric("Nombres diferentes (nom_param)", df["nom_param"].nunique())
    col3.metric("Códigos de unidad diferentes", df["unidades_param"].nunique())
    # 4. TABLA DEL CATALOGO
    st.markdown("---")
    st.header("4. Catálogo de Parámetros Procesado")
    st.dataframe(
        df[[
            "id_parameter",
            "cve_param",
            "nom_param",
            "unidades_param"
        ]].sort_values("id_parameter")
    )
    # 5. GRAFICA 1: PARAMETROS POR CODIGO DE UNIDAD
    st.markdown("---")
    st.header("5. Visualizaciones")
    st.subheader("Cantidad de parámetros por código de unidad")
    conteo_unidades = (
        df["unidades_param"]
        .astype(str)
        .value_counts()
        .sort_values(ascending=False)
    )
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    sns.barplot(x=conteo_unidades.index, y=conteo_unidades.values, ax=ax1)
    ax1.set_title("Cantidad de parametros por codigo de unidad")
    ax1.set_xlabel("Codigo registrado en unidades_param")
    ax1.set_ylabel("Cantidad de parametros")
    plt.xticks(rotation=45)
    st.pyplot(fig1)
    # 6. GRAFICA 2: LISTADO VISUAL DE PARAMETROS
    st.subheader("Listado visual de parámetros")
    df_grafica = df.sort_values("id_parameter")
    fig2, ax2 = plt.subplots(figsize=(10, 11))
    sns.barplot(
        data=df_grafica,
        x="id_parameter",
        y="nom_param",
        orient="h",
        ax=ax2
    )
    ax2.set_title("Catalogo de parametros ambientales")
    ax2.set_xlabel("Identificador del parametro")
    ax2.set_ylabel("Nombre del parametro")
    st.pyplot(fig2)
    # 7. MATRIZ DE CORRELACION
    numericas = df.select_dtypes(include=np.number)
    numericas = numericas.loc[:, numericas.nunique(dropna=True) > 1]
    st.markdown("---")
    st.header("6. Matriz de Correlación")
    if numericas.shape[1] >= 2:
        correlacion = numericas.corr()
        st.dataframe(correlacion)
        fig3, ax3 = plt.subplots(figsize=(7, 5))
        sns.heatmap(
            correlacion,
            annot=True,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax3
        )
        ax3.set_title("Correlacion entre variables numericas")
        st.pyplot(fig3)
    else:
        st.info("No hay suficientes variables numéricas con variación para calcular una correlación útil.")
    # 8. EXPORTAR EL CATALOGO LIMPIO
    st.markdown("---")
    st.header("7. Descargar Resultados")
    
    csv_data = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 Descargar cat_parametros_limpio.csv",
        data=csv_data,
        file_name="cat_parametros_limpio.csv",
        mime="text/csv"
    )
    st.success("¡Análisis exploratorio terminado con éxito!")
else:
    st.info("👆 Por favor, sube tu archivo CSV usando el botón de arriba para comenzar el análisis.")
