import streamlit as st
import pandas as pd
import plotly.express as px
import google.generativeai as genai

# --- Configuration ---
st.set_page_config(layout="wide", page_title="CSV AI Analyzer", page_icon="📊")

# --- Gemini API Configuration ---
# For local development, create a .streamlit/secrets.toml file:
# GEMINI_API_KEY = "YOUR_API_KEY_HERE"
# On Streamlit Cloud, set this as a secret.
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel('gemini-1.5-pro-latest') # Or your preferred Gemini model
except Exception as e:
    st.error(f"Error configuring Gemini API: {e}. Please ensure your API key is set correctly in Streamlit secrets.")
    st.stop()

# --- Helper Functions ---
def get_data_summary(df):
    """Generates a string summary of the DataFrame for the LLM."""
    summary = []
    summary.append(f"The DataFrame has {df.shape[0]} rows and {df.shape[1]} columns.")
    summary.append("\nColumn Names and Data Types:")
    for col in df.columns:
        summary.append(f"- {col} ({df[col].dtype})")
    if not df.empty:
        summary.append("\nFirst 5 rows (sample):")
        summary.append(df.head().to_string())
        summary.append("\nSummary Statistics (for numerical columns):")
        summary.append(df.describe(include='number').to_string())
        summary.append("\nSummary Statistics (for object/categorical columns):")
        summary.append(df.describe(include='object').to_string())
    return "\n".join(summary)

def ask_gemini(data_summary, question):
    """Sends a question and data summary to Gemini and returns the response."""
    if not gemini_model:
        return "Gemini model not initialized. Please check API key."
    try:
        prompt = f"""You are an expert data analysis assistant.
        A user has uploaded a CSV file. Here is a summary of the data:
        --- DATA SUMMARY ---
        {data_summary}
        --- END DATA SUMMARY ---

        The user's question is: "{question}"

        Based on the data summary, please provide a comprehensive and helpful answer.
        If the question is about specific data values you cannot see, explain how the user might find the answer using pandas or by creating a relevant chart.
        If the question is complex, you can suggest Python code (e.g., pandas) to perform the analysis.
        Focus on being helpful and guiding the user.
        """
        response = gemini_model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

# --- Initialize Session State ---
if 'df' not in st.session_state:
    st.session_state.df = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = [] # List of tuples (role, content)
if 'data_summary_for_llm' not in st.session_state:
    st.session_state.data_summary_for_llm = ""

# --- Sidebar for Upload and Controls ---
with st.sidebar:
    st.image("https://streamlit.io/images/brand/streamlit-logo-secondary-colormark-darktext.svg", width=200)
    st.title("📊 CSV AI Analyzer")
    st.markdown("Upload your CSV file and explore your data!")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            st.session_state.data_summary_for_llm = get_data_summary(df)
            st.success("CSV file loaded successfully!")
            st.markdown(f"**File:** `{uploaded_file.name}`")
            st.markdown(f"**Shape:** `{df.shape[0]} rows, {df.shape[1]} columns`")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}")
            st.session_state.df = None
    elif st.session_state.df is not None:
        st.info("Using previously uploaded data.")
        st.markdown(f"**Shape:** `{st.session_state.df.shape[0]} rows, {st.session_state.df.shape[1]} columns`")


    if st.session_state.df is not None:
        st.markdown("---")
        st.header("Display Options")
        if st.checkbox("Show raw data sample", value=False):
            st.dataframe(st.session_state.df.head())
        if st.checkbox("Show data summary statistics", value=False):
            st.write(st.session_state.df.describe(include='all'))

# --- Main Application Area ---
if st.session_state.df is None:
    st.info("👋 Welcome! Please upload a CSV file using the sidebar to get started.")
    st.markdown("""
    **Features:**
    - **AI Chat:** Converse with Gemini about your data.
    - **Dynamic Charts:** Create column, bar, pie, and heatmap charts.
    - **Pivot Tables & Charts:** Generate insightful pivot tables and corresponding visualizations.
    """)
else:
    df = st.session_state.df
    tab1, tab2, tab3 = st.tabs(["💬 AI Chat", "📈 Charts", "📋 Pivot Tables & Charts"])

    # --- AI Chat Tab ---
    with tab1:
        st.subheader("Chat with Gemini about your Data")
        st.markdown(f"Ask questions about the structure, content, or potential insights from your data (`{uploaded_file.name if uploaded_file else 'current data'}`).")

        # Display chat history
        for role, message in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(message)

        user_question = st.chat_input("Ask a question about your data...")

        if user_question:
            # Add user message to chat history
            st.session_state.chat_history.append(("user", user_question))
            with st.chat_message("user"):
                st.markdown(user_question)

            # Get AI response
            with st.spinner("Gemini is thinking..."):
                ai_response = ask_gemini(st.session_state.data_summary_for_llm, user_question)
            
            # Add AI response to chat history
            st.session_state.chat_history.append(("assistant", ai_response))
            with st.chat_message("assistant"):
                st.markdown(ai_response)

    # --- Charts Tab ---
    with tab2:
        st.subheader("Create Visualizations")
        if df.empty:
            st.warning("No data available to create charts.")
        else:
            chart_type = st.selectbox(
                "Select Chart Type",
                ["Column", "Bar", "Pie", "Heatmap (Correlation)"]
            )

            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

            if not numeric_cols and chart_type != "Heatmap (Correlation)": # Heatmap can work with all numerics
                 st.warning("No numeric columns found for typical chart axes (Y-axis, Values). Some charts might be limited.")
            if not categorical_cols and chart_type in ["Column", "Bar", "Pie"]:
                 st.warning("No categorical columns found for typical chart axes (X-axis, Labels). Some charts might be limited.")


            try:
                if chart_type == "Column" or chart_type == "Bar":
                    if not categorical_cols:
                        st.error("Column/Bar charts require at least one categorical column for the X-axis.")
                    elif not numeric_cols:
                        st.error("Column/Bar charts require at least one numeric column for the Y-axis.")
                    else:
                        x_axis = st.selectbox("Select X-axis (Categorical)", categorical_cols)
                        y_axis = st.selectbox("Select Y-axis (Numeric)", numeric_cols)
                        color_axis = st.selectbox("Select Color Dimension (Optional, Categorical)", [None] + categorical_cols)
                        
                        if x_axis and y_axis:
                            title = f"{chart_type} Chart: {y_axis} by {x_axis}"
                            if color_axis:
                                title += f" (colored by {color_axis})"
                            
                            # Aggregate data for better plotting if x_axis has many unique values
                            # For simplicity, directly plotting if categories are not too many.
                            # Or, consider an aggregation step: agg_df = df.groupby(x_axis)[y_axis].sum().reset_index()
                            if chart_type == "Column":
                                fig = px.bar(df, x=x_axis, y=y_axis, color=color_axis, title=title, orientation='v')
                            else: # Bar
                                fig = px.bar(df, x=y_axis, y=x_axis, color=color_axis, title=title, orientation='h') # Swapped for horizontal bar
                            st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Pie":
                    if not categorical_cols:
                        st.error("Pie charts require a categorical column for labels.")
                    elif not numeric_cols:
                        st.error("Pie charts require a numeric column for values.")
                    else:
                        names_col = st.selectbox("Select Labels Column (Categorical)", categorical_cols)
                        values_col = st.selectbox("Select Values Column (Numeric)", numeric_cols)
                        if names_col and values_col:
                            # Aggregate data if necessary to avoid overly complex pie charts
                            agg_df = df.groupby(names_col)[values_col].sum().reset_index()
                            fig = px.pie(agg_df, names=names_col, values=values_col, title=f"Pie Chart: Distribution of {values_col} by {names_col}")
                            st.plotly_chart(fig, use_container_width=True)

                elif chart_type == "Heatmap (Correlation)":
                    if len(numeric_cols) < 2:
                        st.warning("Heatmap (Correlation) requires at least two numeric columns.")
                    else:
                        corr_df = df[numeric_cols].corr()
                        fig = px.imshow(corr_df, text_auto=True, aspect="auto", title="Correlation Heatmap of Numeric Columns")
                        st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error generating chart: {e}")


    # --- Pivot Tables & Charts Tab ---
    with tab3:
        st.subheader("Pivot Tables & Charts")
        if df.empty:
            st.warning("No data available to create pivot tables.")
        else:
            st.markdown("Create a pivot table to summarize your data.")

            # Ensure columns are available
            all_cols = df.columns.tolist()
            if not all_cols:
                st.warning("No columns available in the DataFrame.")
            else:
                index_cols = st.multiselect("Select Row Index(es)", all_cols, default=all_cols[0] if all_cols else None)
                column_cols = st.multiselect("Select Column(s) (Optional)", all_cols)
                
                value_cols_options = numeric_cols if numeric_cols else all_cols # Allow non-numeric for 'count'
                value_cols = st.multiselect("Select Value Column(s)", value_cols_options, default=numeric_cols[0] if numeric_cols else None)
                
                agg_func_options = {
                    "Sum": "sum", "Mean": "mean", "Median": "median",
                    "Min": "min", "Max": "max", "Count": "count",
                    "Standard Deviation": "std", "Variance": "var",
                    "Unique Count": "nunique"
                }
                agg_func_display = st.selectbox("Select Aggregation Function", list(agg_func_options.keys()), index=0)
                selected_agg_func = agg_func_options[agg_func_display]

                if index_cols and value_cols:
                    try:
                        # Handle case where selected value column for count might not be numeric
                        if selected_agg_func == 'count':
                             # For 'count', pandas pivot_table works even with non-numeric value columns if they are specified.
                             # If value_cols is a list of columns, it counts non-NA entries for each.
                             pass
                        elif not all(col in numeric_cols for col in value_cols):
                            st.error(f"Aggregation function '{agg_func_display}' requires numeric value columns. Please select numeric columns for 'Values'.")
                            pivot_df = None
                        else: # All other agg functions typically need numeric values
                            pass


                        pivot_df = pd.pivot_table(
                            df,
                            index=index_cols,
                            columns=column_cols if column_cols else None,
                            values=value_cols,
                            aggfunc=selected_agg_func,
                            fill_value=0 # Fill NaNs with 0 for better display/charting
                        )
                        st.markdown("### Pivot Table Result")
                        st.dataframe(pivot_df)

                        # --- Pivot Chart ---
                        if not pivot_df.empty:
                            st.markdown("### Pivot Chart")
                            # Determine chart type based on pivot table structure
                            # For simplicity, using a bar chart. More complex logic could be added.
                            
                            # If pivot_df has a multi-index for columns, we might need to flatten it or choose specific levels
                            # For now, let's assume a simple pivot or one that px.bar can handle
                            try:
                                if pivot_df.index.nlevels == 1 and (pivot_df.columns.nlevels == 1 or not column_cols):
                                    # Simple case: single index, single or no column dimension
                                    chart_data = pivot_df.reset_index()
                                    melted_chart_data = chart_data.melt(id_vars=chart_data.columns[0], var_name="Category", value_name="Value")
                                    
                                    # Use the first index as x-axis
                                    x_axis_pivot = chart_data.columns[0]
                                    
                                    fig_pivot = px.bar(melted_chart_data, x=x_axis_pivot, y="Value", color="Category",
                                                       title=f"Pivot Chart ({agg_func_display} of {', '.join(value_cols)})",
                                                       barmode='group')
                                    st.plotly_chart(fig_pivot, use_container_width=True)
                                elif pivot_df.index.nlevels > 0 : # Stacked bar for multi-index or complex columns
                                    st.write("Attempting to generate a stacked bar chart for the pivot table.")
                                    # We might need to reset_index and melt for complex pivot tables to make them chartable with px
                                    # This is a simplified approach.
                                    chart_data_stacked = pivot_df.reset_index()
                                    # If columns are multi-index, px might handle it or it might need flattening.
                                    # For now, let's assume px.bar can plot it directly if index is reset.
                                    # The first column after reset_index is usually the primary index.
                                    id_vars_stacked = chart_data_stacked.columns[:pivot_df.index.nlevels].tolist()
                                    value_vars_stacked = chart_data_stacked.columns[pivot_df.index.nlevels:].tolist()

                                    if id_vars_stacked and value_vars_stacked:
                                        melted_stacked_data = chart_data_stacked.melt(
                                            id_vars=id_vars_stacked,
                                            value_vars=value_vars_stacked,
                                            var_name="Pivot Column Category",
                                            value_name="Aggregated Value"
                                        )
                                        
                                        # Use the first index level for x-axis
                                        x_axis_stacked = id_vars_stacked[0]
                                        # Use other index levels for color or facets if needed
                                        color_stacked = id_vars_stacked[1] if len(id_vars_stacked) > 1 else "Pivot Column Category"


                                        fig_pivot_stacked = px.bar(melted_stacked_data, x=x_axis_stacked, y="Aggregated Value",
                                                                   color=color_stacked,
                                                                   hover_data=id_vars_stacked,
                                                                   title=f"Pivot Chart ({agg_func_display} of {', '.join(value_cols)})",
                                                                   barmode='group') # or 'stack'
                                        st.plotly_chart(fig_pivot_stacked, use_container_width=True)
                                    else:
                                        st.info("Cannot automatically generate a chart for this pivot table structure. Consider simplifying the pivot table or using a different charting approach.")

                                else:
                                    st.info("Automatic chart generation for this pivot table structure is complex. Displaying table only.")
                            except Exception as chart_e:
                                st.error(f"Could not generate pivot chart: {chart_e}")
                                st.info("Tip: Simpler pivot tables (e.g., one row index, one value) are easier to chart directly.")

                    except Exception as e:
                        st.error(f"Error creating pivot table: {e}")
                        st.info("Ensure your selections for index, columns, and values are valid for the chosen aggregation function.")

# --- Footer ---
st.markdown("---")
st.markdown("Built with ❤️ by AI for interactive data analysis.")
