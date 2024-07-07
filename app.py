import streamlit as st
import plotly.express as px
import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
import warnings
import uuid

warnings.filterwarnings('ignore')

#----------------------------------------------Page Settings----------------------------------------------------------------
st.set_page_config(page_title="Superstore", page_icon=":bar_chart:", layout="wide")

st.title(":bar_chart: Sample Superstore EDA")
st.markdown('<style>div.block-container{padding-top:2.5rem;}</style>', unsafe_allow_html=True)
#---------------------------------------------------------------------------------------------------------------------------

#----------------------------------------------Dashboard Creation-----------------------------------------------------------
# File Uploader
fl = st.file_uploader(":file_folder: Upload a file", type=["csv", "xlsx", "xls"])
if fl is not None:
    if fl.type == "text/csv":
        df = pd.read_csv(fl, encoding="ISO-8859-1")
    else:
        df = pd.read_excel(fl)
else:

    df = pd.read_csv("sales.csv", encoding="ISO-8859-1")

col1, col2 = st.columns((2))
df["Order Date"] = pd.to_datetime(df["Order Date"], errors='coerce')

# Getting min and max Date
startDate = df["Order Date"].min()
endDate = df["Order Date"].max()

with col1:
    date1 = st.date_input("Start Date", startDate)

with col2:
    date2 = st.date_input("End Date", endDate)

df = df[(df["Order Date"] >= pd.to_datetime(date1)) & (df["Order Date"] <= pd.to_datetime(date2))]

# Create filters for Region, State, and City
st.sidebar.title("Choose your filter 🔽")

region = st.sidebar.multiselect("Pick your Region", df["Region"].unique())
state = st.sidebar.multiselect("Choose your State", df["State"].unique())
city = st.sidebar.multiselect("Choose your City", df["City"].unique())

# Filter the data based on Region, State, and City
filtered_df = df.copy()
if region:
    filtered_df = filtered_df[filtered_df["Region"].isin(region)]
if state:
    filtered_df = filtered_df[filtered_df["State"].isin(state)]
if city:
    filtered_df = filtered_df[filtered_df["City"].isin(city)]

# Category Chart
category_df = filtered_df.groupby(by=["Category"], as_index=False)["Sales"].sum()

with col1:
    st.subheader("Category wise Sales")
    fig = px.bar(category_df, y="Sales", template="seaborn", text=['${:,.2f}'.format(x) for x in category_df["Sales"]])
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Region wise Sales")
    fig = px.pie(filtered_df, values="Sales", names="Region", hole=0.5)
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

cl1, cl2 = st.columns((2))

with cl1:
    with st.expander("Category_ViewData &#x2935;"):
        st.write(category_df.style.background_gradient(cmap="Blues"))
        csv = category_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name='Category.csv', mime="text/csv", help="Click here to download the CSV file")

with cl2:
    with st.expander("Region_ViewData &#x2935;"):
        region_df = filtered_df.groupby(by=["Region"], as_index=False)["Sales"].sum()
        st.write(region_df.style.background_gradient(cmap="Blues"))
        csv = region_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Data", data=csv, file_name="Region.csv", mime="text/csv", help="Click here to download the CSV file")

# Time Series Analysis with Datetime Data
filtered_df["month_year"] = filtered_df["Order Date"].dt.to_period("M")
st.subheader("Time Series Analysis")

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y-%b"))["Sales"].sum()).reset_index()
fig2 = px.line(linechart, x="month_year", y="Sales", labels={"Sales": "Amount"}, height=500, template="gridon")
st.plotly_chart(fig2, use_container_width=True)

with st.expander("View Data of TimeSeries: &#x2935;"):
    st.write(linechart.T.style.background_gradient(cmap="Blues"))
    csv = linechart.to_csv(index=False).encode('utf-8')
    st.download_button("Download Data", data=csv, file_name="TimeSeries.csv", mime="text/csv", help="Click here to download the CSV file")

# Create a Tree Map based on Region, Category, Sub-category
st.subheader("Hierarchical view of Sales using Tree Map")

fig3 = px.treemap(filtered_df, path=["Region", "Category", "Sub-Category"], values="Sales", hover_data=["Sales"], color="Sub-Category")
fig3.update_layout(height=650, width=800)
st.plotly_chart(fig3, use_container_width=True)

# Two Pie Charts for Category and Segment wise sales
chart1, chart2 = st.columns((2))

with chart1:
    st.subheader('Segment wise sales')
    fig = px.pie(filtered_df, values="Sales", names="Segment", template="plotly_dark")
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

with chart2:
    st.subheader('Category wise sales')
    fig = px.pie(filtered_df, values="Sales", names="Category", template="plotly_dark")
    fig.update_traces(textposition="inside")
    st.plotly_chart(fig, use_container_width=True)

# Creating customized Table
import plotly.figure_factory as ff
st.subheader(":point_right: Month wise Sub-category Sales Summary")
with st.expander("Summary_Table &#x2935;"):
    df_sample = df[:5][["Region", "State", "City", "Category", "Sales", "Profit", "Quantity"]]
    fig = ff.create_table(df_sample, colorscale="Cividis")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("Month wise Sub-Category Table")
    filtered_df["month"] = filtered_df["Order Date"].dt.month_name()
    sub_category_year = pd.pivot_table(data=filtered_df, values="Sales", index="Sub-Category", columns="month")
    st.write(sub_category_year.style.background_gradient(cmap="Blues"))

# Create a Scatter Plot
data1 = px.scatter(filtered_df, x="Sales", y="Profit", size="Quantity")
data1.update_layout(title="Relationship between sales and profit using scatter plot", titlefont=dict(size=20), xaxis=dict(title="Sales", titlefont=dict(size=19)), yaxis=dict(title="Profit", titlefont=dict(size=19)))
st.plotly_chart(data1, use_container_width=True)

# Overall view and download
with st.expander("View Data &#x2935;"):
    st.write(filtered_df.iloc[:500, 1:20:2].style.background_gradient(cmap="Oranges"))

csv = df.to_csv(index=False).encode('utf-8')
st.download_button('Download Data', data=csv, file_name="Dataset.csv", mime="text/csv", help="Download the dataset from here")

# Chat with dataframe
st.sidebar.title("Chat with Data 💬 ")

# Retrieve API key from Streamlit secrets
API = st.secrets["secrets"]["api_key"]

# Model selection
model = st.sidebar.selectbox(
    'Choose a model', ['Llama3-8b-8192', 'Llama3-70b-8192', 'Mixtral-8x7b-32768', 'Gemma-7b-It']
)

#---------------------------------------------Model configuration-----------------------------------------------------------

# Toggle Data for use
st.sidebar.header("Options")
show_data = st.sidebar.toggle("Use raw data", value=False, help="Toggle to choose whether to work with filtered Data or Raw Data")
if show_data:
    filtered_df = df.copy()
    st.subheader("Superstore Data")
    st.dataframe(filtered_df.head(20))

# Initialize the language model
llm = ChatGroq(model_name=model, api_key=API)

# Load Data
smart_df = SmartDataframe(filtered_df, config={"llm": llm})

# Streamlit Interface
st.header(":point_right: AI-Powered Data Discovery 🤖")

# Initialize session state for question history
if 'question_history' not in st.session_state:
    st.session_state.question_history = []

# Display chat messages from history on app return
for message in st.session_state.question_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "image_path" in message:
            st.image(message["image_path"], caption="Sample Chart", use_column_width=True)

if chat_input := st.chat_input("Ask a question about the data"):
    try:
        with st.chat_message("user"):
            st.markdown(chat_input)

        # Display prompt in the history from the user
        st.session_state.question_history.append({"role": "user", "content": chat_input})
        
        answer = smart_df.chat(chat_input)

        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Display answer in the history from the assistant
        answer_entry = {"role": "assistant", "content": answer}

        # Display the image if it exists
        image_path = f"exports/charts/temp_chart_{uuid.uuid4()}.png"
        if os.path.exists("exports/charts/temp_chart.png"):
            os.rename("exports/charts/temp_chart.png", image_path)
            st.image(image_path, caption="Sample Chart", use_column_width=True)
            answer_entry["image_path"] = image_path

        st.session_state.question_history.append(answer_entry)

    except Exception as e:
        st.error(f"Error occurred: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Built with ❤️ by Gaurav Yadav")
