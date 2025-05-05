import streamlit as st
import requests
import json
import os
import pandas as pd
from recommend_engine import SHLRecommendationEngine
from evaluator import RecommendationEvaluator
import base64
from PIL import Image
import io
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set page config
st.set_page_config(
    page_title="SHL Assessment Recommendation System",
    page_icon="📊",
    layout="wide",
)

# Initialize recommendation engine
@st.cache_resource
def get_recommendation_engine():
    return SHLRecommendationEngine()

engine = get_recommendation_engine()

# Custom CSS for blue and purple theme
def add_custom_css():
    st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #0b0b3b 0%, #30124e 100%);
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #0b0b3b 0%, #30124e 100%);
    }
    .stTextInput > label, .stSelectbox > label {
        color: white !important;
    }
    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    .stTextArea > div > div > textarea {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    .stSelectbox > div > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    .stTabs > div > div > div > div {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
    }
    .stDataFrame {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stButton > button {
        background-color: #6c63ff !important;
        color: white !important;
        border: none !important;
        border-radius: 5px !important;
        padding: 0.5rem 1rem !important;
        font-weight: bold !important;
    }
    .stButton > button:hover {
        background-color: #5046e5 !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: rgba(255, 255, 255, 0.1) !important;
        color: white !important;
        border-radius: 4px 4px 0px 0px;
        padding: 10px 20px;
        border: none;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(108, 99, 255, 0.7) !important;
        color: white !important;
        border-bottom: 2px solid #6c63ff;
    }
    .css-145kmo2 {
        color: white !important;
    }
    .stDataFrame tbody tr:nth-child(odd) {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    .stDataFrame tbody tr:nth-child(even) {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    .stDataFrame tbody tr:hover {
        background-color: rgba(108, 99, 255, 0.3) !important;
    }
    .stDataFrame thead th {
        background-color: rgba(108, 99, 255, 0.5) !important;
        color: white !important;
    }
    .stDataFrame td {
        color: white !important;
    }
    .css-1n76uvr {
        color: white !important;
    }
    .stMarkdown p {
        color: white !important;
    }
    .stExpander {
        background-color: rgba(255, 255, 255, 0.1) !important;
        border-radius: 5px !important;
    }
    .stExpander > details > summary {
        color: white !important;
        font-weight: bold !important;
    }
    .stRadio > div > label {
        color: white !important;
    }
    .stCheckbox > div > label {
        color: white !important;
    }
    .stAlert > div {
        background-color: rgba(108, 99, 255, 0.2) !important;
        border: 1px solid #6c63ff !important;
    }
    .css-18ni7ap {
        background-color: rgba(255, 255, 255, 0.1) !important;
    }
    div[data-testid="stVerticalBlock"] {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    .stSidebar {
        background-color: rgba(0, 0, 0, 0.3) !important;
    }
    div[data-testid="stSidebar"] > div > div > div > div {
        background-color: rgba(0, 0, 0, 0) !important;
    }
    .stSidebar .stButton > button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# Add custom CSS
add_custom_css()

# Create header with logo
def create_header():
    col1, col2 = st.columns([1, 5])
    
    with col1:
        # SHL-like logo placeholder (blue/purple gradient)
        logo_html = f"""
        <div style="
            width: 80px;
            height: 80px;
            background: linear-gradient(45deg, #6c63ff, #3023ae);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-weight: bold;
            font-size: 24px;
            margin: 10px;
        ">
            SHL
        </div>
        """
        st.markdown(logo_html, unsafe_allow_html=True)
    
    with col2:
        st.title("SHL Assessment Recommendation System")
        st.markdown("""
        <p style="font-size: 1.2em; margin-top: -10px;">
            Find the perfect assessment for your hiring needs
        </p>
        """, unsafe_allow_html=True)

# Create header
create_header()

# Create tabs
tab1, tab2, tab3 = st.tabs(["🔍 Get Recommendations", "📊 Evaluation", "ℹ️ About"])

# Tab 1: Recommendations
with tab1:
    st.markdown("""
    <h2 style="margin-top: 10px;">Find the Right Assessment</h2>
    <p>Enter a job description or query to get tailored assessment recommendations.</p>
    """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Input options
        input_type = st.radio("Select input type:", ["Text Input", "URL Input"], horizontal=True)
        
        if input_type == "Text Input":
            user_query = st.text_area("Enter job description or query:", 
                                     height=150,
                                     placeholder="Example: I am hiring for Java developers who can collaborate effectively with business teams. Looking for an assessment that can be completed in 40 minutes.")
            url_input = None
        else:
            url_input = st.text_input("Enter job description URL:", 
                                    placeholder="https://example.com/job-description")
            user_query = None
            
    with col2:
        st.markdown("<br><br>", unsafe_allow_html=True)
        max_results = st.slider("Maximum results:", min_value=1, max_value=10, value=5)
        
        # Additional filters
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### Optional Filters")
        
        duration_filter = st.slider("Max Duration (minutes):", 
                                  min_value=0, max_value=120, value=60, step=5)
        
        col1, col2 = st.columns(2)
        with col1:
            remote_testing = st.checkbox("Remote Testing")
        with col2:
            adaptive_testing = st.checkbox("Adaptive Testing")
    
    # Submit button    
    if st.button("Get Recommendations", use_container_width=True):
        if not user_query and not url_input:
            st.error("Please enter a query or URL")
        else:
            with st.spinner("Finding the best assessments for you..."):
                try:
                    # Process the query
                    if url_input:
                        # Use URL content as query
                        import requests
                        from bs4 import BeautifulSoup
                        
                        response = requests.get(url_input, timeout=10)
                        soup = BeautifulSoup(response.text, 'html.parser')
                        query = soup.get_text(separator=" ", strip=True)
                    else:
                        query = user_query
                    
                    # Get recommendations
                    recommendations = engine.recommend(query, top_k=max_results * 2)  # Get more for filtering
                    
                    # Apply filters if specified
                    filters = {}
                    if duration_filter > 0:
                        filters["duration_limit"] = duration_filter
                    if remote_testing:
                        filters["remote_testing"] = True
                    if adaptive_testing:
                        filters["adaptive_testing"] = True
                    
                    filtered_recommendations = engine.filter_recommendations(recommendations, **filters)
                    
                    # Use at most max_results
                    filtered_recommendations = filtered_recommendations[:max_results]
                    
                    if not filtered_recommendations:
                        st.warning("No assessments match your criteria. Try adjusting your filters.")
                        
                        # Show some recommendations without filters
                        st.markdown("### Recommendations Without Filters")
                        basic_recommendations = recommendations[:max_results]
                        
                        # Create DataFrame
                        df = pd.DataFrame([
                            {
                                "Assessment": f"[{rec['title']}]({rec['url']})",
                                "Remote Testing": rec['remote_testing_support'],
                                "Adaptive Testing": rec['adaptive_irt_support'],
                                "Duration": rec['duration'],
                                "Test Type": rec['test_type']
                            }
                            for rec in basic_recommendations
                        ])
                        
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        # Show recommendations
                        st.markdown("### Recommended Assessments")
                        
                        # Create DataFrame
                        df = pd.DataFrame([
                            {
                                "Assessment": f"[{rec['title']}]({rec['url']})",
                                "Remote Testing": rec['remote_testing_support'],
                                "Adaptive Testing": rec['adaptive_irt_support'],
                                "Duration": rec['duration'],
                                "Test Type": rec['test_type']
                            }
                            for rec in filtered_recommendations
                        ])
                        
                        st.dataframe(df, hide_index=True, use_container_width=True)
                        
                        # Show explanations
                        st.markdown("### Why These Assessments?")
                        for i, rec in enumerate(filtered_recommendations[:3], 1):
                            with st.expander(f"Why recommend: {rec['title']}"):
                                st.markdown(f"""
                                * **Test Type**: {rec['test_type']}
                                * **Duration**: {rec['duration']}
                                * **Remote Testing**: {rec['remote_testing_support']}
                                * **Adaptive Testing**: {rec['adaptive_irt_support']}
                                
                                This assessment was recommended based on the skills and requirements mentioned in your query.
                                """)
                
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

# Tab 2: Evaluation
with tab2:
    st.markdown("""
    <h2 style="margin-top: 10px;">System Evaluation</h2>
    <p>View the performance metrics of the recommendation system.</p>
    """, unsafe_allow_html=True)
    
    # Initialize evaluator
    evaluator = RecommendationEvaluator(recommendation_engine=engine)
    
    # Add k-value selection
    col1, col2 = st.columns(2)
    with col1:
        k_values = st.multiselect(
            "Select k values for evaluation:",
            options=[1, 3, 5, 7, 10],
            default=[3, 5, 10]
        )
    
    with col2:
        verbose_output = st.checkbox("Show detailed output", value=False)
    
    if st.button("Run Evaluation", key="run_eval"):
        if not k_values:
            st.error("Please select at least one k value for evaluation")
        else:
            with st.spinner("Running evaluation..."):
                # Run evaluation with selected k values
                results = evaluator.evaluate(k_values=k_values, verbose=verbose_output)
                
                # Display metrics
                st.markdown("### Overall Evaluation Metrics")
                
                # Create two columns for metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### Mean Average Precision (MAP)")
                    map_data = {
                        'K value': k_values,
                        'MAP': [results["overall"][f"map@{k}"] for k in k_values]
                    }
                    map_df = pd.DataFrame(map_data)
                    st.dataframe(map_df, hide_index=True, use_container_width=True)
                    
                    # Create MAP chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar([f'MAP@{k}' for k in k_values], 
                           [results["overall"][f"map@{k}"] for k in k_values],
                           color='#6c63ff')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Score')
                    ax.set_title('Mean Average Precision (MAP)')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Set the figure background to be transparent
                    fig.patch.set_alpha(0)
                    ax.set_facecolor('none')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')
                    
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("#### Mean Recall")
                    recall_data = {
                        'K value': k_values,
                        'Mean Recall': [results["overall"][f"mean_recall@{k}"] for k in k_values]
                    }
                    recall_df = pd.DataFrame(recall_data)
                    st.dataframe(recall_df, hide_index=True, use_container_width=True)
                    
                    # Create Mean Recall chart
                    fig, ax = plt.subplots(figsize=(8, 4))
                    ax.bar([f'Recall@{k}' for k in k_values], 
                           [results["overall"][f"mean_recall@{k}"] for k in k_values],
                           color='#30124e')
                    ax.set_ylim(0, 1)
                    ax.set_ylabel('Score')
                    ax.set_title('Mean Recall')
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    
                    # Set the figure background to be transparent
                    fig.patch.set_alpha(0)
                    ax.set_facecolor('none')
                    ax.tick_params(axis='x', colors='white')
                    ax.tick_params(axis='y', colors='white')
                    ax.yaxis.label.set_color('white')
                    ax.xaxis.label.set_color('white')
                    ax.title.set_color('white')
                    
                    st.pyplot(fig)
                
                # Show precision metrics
                st.markdown("#### Mean Precision")
                precision_data = {
                    'K value': k_values,
                    'Mean Precision': [results["overall"][f"mean_precision@{k}"] for k in k_values]
                }
                precision_df = pd.DataFrame(precision_data)
                st.dataframe(precision_df, hide_index=True, use_container_width=True)
                
                # Create Mean Precision chart
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.bar([f'Precision@{k}' for k in k_values], 
                       [results["overall"][f"mean_precision@{k}"] for k in k_values],
                       color='#4c71b6')
                ax.set_ylim(0, 1)
                ax.set_ylabel('Score')
                ax.set_title('Mean Precision')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                
                # Set the figure background to be transparent
                fig.patch.set_alpha(0)
                ax.set_facecolor('none')
                ax.tick_params(axis='x', colors='white')
                ax.tick_params(axis='y', colors='white')
                ax.yaxis.label.set_color('white')
                ax.xaxis.label.set_color('white')
                ax.title.set_color('white')
                
                st.pyplot(fig)
                
                # Detailed evaluation for specific k
                st.markdown("### Per-Query Evaluation")
                
                selected_k = st.selectbox("Select k for detailed query evaluation:", options=k_values)
                detailed_results = evaluator.detailed_evaluation(k=selected_k) if hasattr(evaluator, 'detailed_evaluation') else None
                
                if detailed_results:
                    for i, result in enumerate(detailed_results):
                        with st.expander(f"Query {i+1}: {result['query'][:100]}..."):
                            st.markdown(f"""
                            * **Recall@{selected_k}**: {result['recall@' + str(selected_k)]:.4f}
                            * **Precision@{selected_k}**: {result['precision@' + str(selected_k)]:.4f}
                            * **AP@{selected_k}**: {result['ap@' + str(selected_k)]:.4f}
                            
                            **Recommendations:**
                            """)
                            
                            for j, rec in enumerate(result["recommendations"], 1):
                                st.markdown(f"{j}. [{rec['title']}]({rec['url']})")
                            
                            st.markdown("**Relevant Items in Test Set:**")
                            for j, item in enumerate(result["relevant_items"], 1):
                                st.markdown(f"{j}. {item}")
                else:
                    # Extract per-query results from the results dictionary
                    if "per_query" in results:
                        for i, (query, query_result) in enumerate(results["per_query"].items(), 1):
                            query_short = query[:100] + "..." if len(query) > 100 else query
                            with st.expander(f"Query {i}: {query_short}"):
                                st.markdown(f"""
                                * **Recall@{selected_k}**: {query_result['metrics'].get(f'recall@{selected_k}', 0):.4f}
                                * **Precision@{selected_k}**: {query_result['metrics'].get(f'precision@{selected_k}', 0):.4f}
                                * **AP@{selected_k}**: {query_result['metrics'].get(f'ap@{selected_k}', 0):.4f}
                                
                                **Recommendations:**
                                """)
                                
                                for j, title in enumerate(query_result["recommended_titles"][:selected_k], 1):
                                    # Format with URL if available
                                    if isinstance(query_result["recommendations"][j-1], dict) and "url" in query_result["recommendations"][j-1]:
                                        url = query_result["recommendations"][j-1]["url"]
                                        st.markdown(f"{j}. [{title}]({url})")
                                    else:
                                        st.markdown(f"{j}. {title}")
                                
                                st.markdown("**Relevant Items in Test Set:**")
                                for j, item in enumerate(query_result["relevant_items"], 1):
                                    st.markdown(f"{j}. {item}")

# Tab 3: About
with tab3:
    st.markdown("""
    <h2 style="margin-top: 10px;">About This System</h2>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    ### SHL Assessment Recommendation System
    
    This application helps hiring managers find the right assessments for their hiring needs. It uses natural language processing and retrieval-augmented generation (RAG) to analyze job descriptions and recommend the most relevant SHL assessments.
    
    #### Key Features:
    
    - **Natural Language Processing**: Understand job descriptions and requirements
    - **Smart Filtering**: Filter assessments by duration, remote testing support, and more
    - **Real-time Recommendations**: Get instant assessment recommendations
    - **Performance Evaluation**: System evaluated using standard retrieval metrics
    
    #### How It Works:
    
    1. **Data Collection**: Assessment data is collected from the SHL product catalog
    2. **Embedding Generation**: Advanced language models create deep representations of assessments
    3. **Semantic Matching**: Your query is matched with the most relevant assessments
    4. **Filtering & Ranking**: Results are filtered and ranked based on relevance and criteria
    
    #### Technology Stack:
    
    - **Backend**: Python with FastAPI
    - **Embedding Models**: Sentence Transformers & HuggingFace
    - **Vector Search**: FAISS for efficient similarity search
    - **Web Interface**: Streamlit with custom styling
    """)
    
    # Architecture diagram - replaced with image upload
    st.markdown("### System Architecture")
    
    # Image upload section
    uploaded_image = st.file_uploader("Upload architecture diagram", type=["png", "jpg", "jpeg"])
    
    if uploaded_image is not None:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="System Architecture Diagram", use_column_width=True)
    else:
        # Display a message when no image is uploaded
        st.info("Upload an architecture diagram to visualize the system components")
        
        # Optional: Display a placeholder text diagram
        st.markdown("""
        <div style="background-color: rgba(255, 255, 255, 0.1); padding: 20px; border-radius: 10px;">
            <pre style="color: white; font-family: monospace; overflow-x: auto;">
        [Upload an architecture diagram to replace this placeholder]
            </pre>
        </div>
        """, unsafe_allow_html=True)
    
    # API information
    st.markdown("### API Integration")
    st.markdown("""
    The system provides a REST API for integration with other applications:
    
    **Base URL**: `https://your-deployment-url.com/api/v1`
    
    **Endpoints**:
    - `GET /health` - Check API status
    - `POST /recommend` - Get assessment recommendations
    
    **Example Request**:
    ```json
    {
        "query": "I am hiring for Java developers who can collaborate effectively with business teams. Looking for an assessment that can be completed in 40 minutes.",
        "max_results": 5
    }
    ```
    
    **Example Response**:
    ```json
    {
        "status": "success",
        "recommendations": [
            {
                "title": "Core Java (Entry Level) (New) | SHL",
                "url": "https://www.shl.com/solutions/products/product-catalog/view/core-java-entry-level-new/",
                "remote_testing_support": "Yes",
                "adaptive_irt_support": "No",
                "duration": "40 minutes",
                "test_type": "Technical Assessment"
            },
            ...
        ]
    }
    ```
    """)
    
    # Contact info
    st.markdown("### Need Help?")
    st.markdown("""
    For support or more information, please contact us at:
    
    📧 roybishal362@gmail.com
    """)