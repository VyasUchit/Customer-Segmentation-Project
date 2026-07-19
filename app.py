import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import joblib  

# Load Model :
kmeans = joblib.load("customer_segmentation_model.pkl")
scaler = joblib.load("scaler.pkl")

if 'page' not in st.session_state:
    st.session_state.page = "Home"

# Side bar Navigation

st.sidebar.title("📌 Navigation")

if st.sidebar.button("🏠 Home"):
    st.session_state.page = "Home"

if st.sidebar.button("📊 Dashboard"):
    st.session_state.page = "Dashboard"

if st.sidebar.button("🎯 Customer_Segemntation"):
    st.session_state.page = "Customer_Segmen"

if st.sidebar.button("💼 Business_insights"):
    st.session_state.page = "Business_insights" 

# Page - 1 : Home Page
if st.session_state.page == "Home":
    st.title("🎯 AI-Powered Customer Segmentation System")

    st.set_page_config(
        page_title="About Project",
        page_icon="👨‍💻",
        layout="wide"
    )

    st.title("👨‍💻 About Project")

    st.write("""
    This project applies Machine Learning to segment customers based on their purchasing behaviour.
    By identifying different customer groups, businesses can design targeted marketing strategies,
    improve customer retention, and increase overall sales.
    """)

    st.divider()

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric("👥 Total Customers", "2,236")
    with col2:
        st.metric("📊 Features", "29")
    with col3:
        st.metric("🎯 Clusters", "3")
    with col4:
        st.metric("🤖 Algorithm", "K-Means")

    st.header("Customer Segmentation Analysis")

    st.write("""
    Businesses often treat every customer the same.
    This project groups customers based on their purchasing behavior so that companies can provide personalized marketing strategies.
    """)

    st.header("📌 Project Overview")

    st.info("""
    The Customer Segmentation System uses the **K-Means Clustering** algorithm to divide customers into
    three meaningful segments based on purchasing behaviour.

    The project includes data preprocessing, feature scaling, clustering,
    interactive dashboards, customer segment prediction, and business insights
    using Streamlit.
    """)

    st.divider()

    st.header("⚙️ Project Workflow")

    st.markdown("""
    1. Load Customer Dataset

    2. Data Cleaning & Preprocessing

    3. Exploratory Data Analysis (EDA)

    4. Feature Scaling using StandardScaler

    5. Train K-Means Clustering Model

    6. Save Model & Scaler

    7. Predict Customer Segment

    8. Generate Business Insights

    9. Deploy Interactive Streamlit Application
    """)

    st.divider()

    st.header("🛠️ Technologies Used")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.success("""
    Python

    Pandas

    NumPy
    """)

    with col2:
        st.success("""
    Matplotlib

    Seaborn

    Scikit-learn
    """)

    with col3:
        st.success("""
    Streamlit

    Joblib

    Jupyter Notebook
    """)

    st.divider()

    st.header("🤖 Machine Learning Model")

    st.write("""
    "Algorithm:" K-Means Clustering

    "Number of Clusters:" 3

    "Evaluation Metric:" Silhouette Score

    "Purpose:" Customer Segmentation based on purchasing behaviour.
    """)

    st.divider()

    st.header("✨ Project Features")

    st.markdown("""
    ✅ Interactive Dashboard

    ✅ Customer Segment Prediction

    ✅ Business Insights

    ✅ Data Visualization

    ✅ Marketing Recommendations

    ✅ User-Friendly Streamlit Interface
    """)

    st.header("🚀 Future Enhancements")

    st.markdown("""
    - Real-time customer data integration

    - Interactive business reports

    - Download prediction reports

    - Advanced customer analytics

    - Cloud deployment
    """)

    st.divider()

    st.header("👨‍💻 Developer")

    st.success("""
    Name: Uchit Vyas

    Role: Information Technology Student

    Interests: Data Science, Machine Learning, Data Analytics and Artificial Intelligence.
    """)


# Page 2 - Dashboard

if st.session_state.page == "Dashboard":
    st.title("📊 Customer Segmentation Dashboard")

    st.write("Analyze customer behavior across different customer segments.")

    df = pd.read_csv(r"customer_segmentation.csv")


    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(label="👥 Total Customers", value=df.shape[0])

    with col2:
        st.metric(label="🎯 Total Clusters", value=df['Cluster'].nunique())

    with col3:
        st.metric(label="💰 Average Income", value=f"₹{df['Income'].mean():,.0f}")


    cluster_name = {
        0 : 'Premium',
        1 : 'Regular',
        2 : 'Low Value'
    }

    df["Customer Segment"] = df["Cluster"].map(cluster_name)
    fig = px.pie(
        df,
        names="Customer Segment",
        title="Customer Distribution"
    )

    st.plotly_chart(fig,use_container_width=True)

    fig = px.bar(
        df["Customer Segment"].value_counts().reset_index(),
        x = "Customer Segment",
        y = 'count',
        title = 'Customers  in each Segment'
    )
    st.plotly_chart(fig,use_container_width=True)

    df["Total_spending"] = (
    df["MntWines"] +
    df["MntFruits"] +
    df["MntMeatProducts"] +
    df["MntFishProducts"] +
    df["MntSweetProducts"] +
    df["MntGoldProds"]
    )

    fig = px.scatter(
    df,
    x="Income",
    y="Total_spending",
    color="Customer Segment",
    title="Income vs Total Spending",
    hover_data=["Recency", "NumWebPurchases", "NumStorePurchases"]
    )

    st.plotly_chart(fig, use_container_width=True)


    fig = px.box(
        df,
        x="Customer Segment",
        y="Income",
        color="Customer Segment",
        title="Income Distribution by Segment"
    )

    st.plotly_chart(fig,use_container_width=True)

    important_cols = [
    "Income",
    "Recency",
    "MntWines",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "NumWebPurchases",
    "NumStorePurchases",
    "Cluster"
    ]

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        df[important_cols].corr(),
        annot=True,
        cmap="Blues",
        fmt=".2f"
    )

    plt.title("Correlation Heatmap")

    st.pyplot(plt)

# Page 3 - Customer_segment

if st.session_state.page == "Customer_Segmen":
    st. title("🎯 Customer Segment Predictor")

    st.write(
        "Predict the customer segment based on customer purchasing behaviour using the trained K-Means clustering model."
    )

    st.info("""
    ### 📌 How to Use

    1. Enter the customer purchase details.
    2. Click Predict Customer Segment.
    3. The trained K-Means model will predict the customer segment.
    4. Business recommendations will be displayed.
    """)

    col1, col2 = st.columns(2)
    with col1:

        income = st.number_input(
            "Income",
            min_value=0.0,
            value=50000.0,
            help="Enter the customer's annual income."
        )

        recency = st.number_input(
            "Recency",
            min_value=0,
            value=30,
            help="Number of days since the customer's last purchase. Lower values indicate a more recent purchase."
        )

        beverage = st.number_input(
            "Beverage Spending",
            min_value=0.0,
            value=100.0,
            help="Total amount spent on beverages."
        )

        food = st.number_input(
            "Food Spending",
            min_value=0.0,
            value=100.0,
            help="Total amount spent on food products."
        )

    with col2:

        fresh = st.number_input(
            "Fresh Products Spending",
            min_value=0.0,
            value=50.0,
            help="Total amount spent on fresh products."
        )

        snacks = st.number_input(
            "Snacks & Sweets Spending",
            min_value=0.0,
            value=30.0,
            help="Total amount spent on snacks and sweet products."
        )

        online = st.number_input(
            "Online Purchases",
            min_value=0,
            value=5,
            help="Number of purchases made through the online store."
        )

        store = st.number_input(
            "In-Store Purchases",
            min_value=0,
            value=5,
            help="Number of purchases made from physical retail stores."
        )
    st.divider()

    if st.button("🎯 Predict Customer Segment", use_container_width=True):

        input_df = pd.DataFrame({
            "Income": [income],
            "Recency": [recency],
            "MntWines": [beverage],
            "MntMeatProducts": [food],
            "MntFishProducts": [fresh],
            "MntSweetProducts": [snacks],
            "NumWebPurchases": [online],
            "NumStorePurchases": [store]
        })
        input_scaled = scaler.transform(input_df)

        cluster = kmeans.predict(input_scaled)[0]

        st.divider()

        st.write("Scaled Input:")
        st.write(input_scaled)

        cluster = kmeans.predict(input_scaled)[0]

        st.write("Predicted Cluster:", cluster)

        st.subheader("📊 Prediction Result")

        if cluster == 1:

            st.success("🌟 Premium Customer")

            st.markdown("""
    ### Customer Characteristics

    - High purchasing power
    - Frequently purchases premium products
    - Loyal customer
    - High business value

    ### Business Recommendation

    ✅ VIP Membership

    ✅ Personalized Product Recommendations

    ✅ Premium Discounts

    ✅ Early Access to New Products

    ✅ Exclusive Loyalty Rewards
    """)

        elif cluster == 2:

            st.warning("🟡 Regular Customer")

            st.markdown("""
    ### Customer Characteristics

    - Moderate spending
    - Regular purchases
    - Good growth potential

    ### Business Recommendation

    ✅ Loyalty Reward Program

    ✅ Cross Selling

    ✅ Bundle Offers

    ✅ Seasonal Discounts

    ✅ Personalized Email Marketing
    """)

        else:

            st.error("🔴 Low Value Customer")

            st.markdown("""
    ### Customer Characteristics

    - Low spending
    - Infrequent purchases
    - High risk of inactivity

    ### Business Recommendation

    ✅ Discount Coupons

    ✅ Win-back Campaigns

    ✅ Limited Time Offers

    ✅ Promotional Emails

    ✅ Customer Re-engagement Strategy
    """)
            
# Page - 4 - Business_insights

if st.session_state.page == "Business_insights":
    st.title("💼 Business Insights")

    st.info("""
    This page summarizes the key business insights derived from the K-Means customer segmentation model.
    Each customer segment represents a unique purchasing behavior, allowing businesses to design targeted marketing strategies and improve customer retention.
    """)

    st.divider()

    # Premium Customers
    with st.expander("🌟 Premium Customers", expanded=True):

        st.subheader("📌 Characteristics")
        st.markdown("""
    - High Income
    - High Spending
    - Frequent Purchases
    - Loyal Customers
    - High Customer Engagement
    """)

        st.subheader("📈 Business Insights")

        st.write("""
    Premium customers are the company's most valuable customers.
    They contribute a large portion of the company's revenue through
    frequent purchases and higher spending.
    Maintaining strong relationships with these customers is essential
    for long-term business growth.
        """)

        st.subheader("💡 Business Recommendations")

        st.success("""
    ✅ Offer VIP Membership

    ✅ Personalized Product Recommendations

    ✅ Early Access to New Products

    ✅ Premium Customer Support

    ✅ Exclusive Discounts & Reward Programs
        """)

    st.divider()

    # Regular Customers
    with st.expander("🟡 Regular Customers"):

        st.subheader("📌 Characteristics")

        st.markdown("""
    - Moderate Income
    - Moderate Spending
    - Average Purchase Frequency
    - Consistent Shopping Behaviour
        """)

        st.subheader("📈 Business Insights")

        st.write("""
    Regular customers contribute consistently to overall sales.
    With proper engagement strategies, they have the potential to become
    premium customers in the future.
        """)

        st.subheader("💡 Business Recommendations")

        st.success("""
    ✅ Loyalty Programs

    ✅ Cross-selling Products

    ✅ Reward Points

    ✅ Personalized Email Campaigns

    ✅ Product Recommendations
        """)

    st.divider()

    # Low Value Customers
    with st.expander("🔴 Low-Value Customers"):

        st.subheader("📌 Characteristics")

        st.markdown("""
    - Low Income
    - Low Spending
    - Less Frequent Purchases
    - Lower Customer Engagement
        """)

        st.subheader("📈 Business Insights")

        st.write("""
    Low-value customers contribute less revenue and interact less
    frequently with the business. Promotional campaigns can help
    increase their engagement and encourage repeat purchases.
        """)

        st.subheader("💡 Business Recommendations")

        st.success("""
    ✅ Discount Coupons

    ✅ Seasonal Offers

    ✅ Bundle Discounts

    ✅ Promotional Email Campaigns

    ✅ Re-engagement Marketing
        """)

    st.divider()


    st.header("🎯 Business Impact")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("🌟 Premium", "High Revenue")

    with col2:
        st.metric("🟡 Regular", "Growth Potential")

    with col3:
        st.metric("🔴 Low-Value", "Re-engagement Needed")

        st.divider()

    st.success("""
    ### 🎯 Final Conclusion

    Customer segmentation helps businesses understand different customer behaviors and make data-driven marketing decisions.

    By targeting Premium, Regular, and Low-Value customers with different strategies, companies can improve customer satisfaction, increase sales, and maximize long-term profitability.
    """)