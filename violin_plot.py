import pandas as pd
import plotly.express as px
import logging
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def clean_data(df):
    """Clean and preprocess the data"""
    try:
        # Replace empty strings and '?' with NaN
        df = df.replace(['', '?'], pd.NA)
        
        # Convert numerical columns to float
        numerical_columns = [
            'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
            'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
        ]
        
        for col in numerical_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with NaN in numerical features
        df = df.dropna(subset=numerical_columns)
        
        return df
    except Exception as e:
        logging.error(f"Error in data cleaning: {str(e)}")
        raise

def create_violin_plot(df, column):
    """Create an interactive violin plot for a specific column"""
    try:
        fig = px.violin(
            df,
            y=column,
            x="classification",
            color="classification",
            box=True,
            template='plotly_dark',
            title=f'Distribution of {column} by Classification'
        )
        return fig
    except Exception as e:
        logging.error(f"Error creating violin plot for {column}: {str(e)}")
        raise

def save_plot(fig, filename):
    """Save the plot to HTML file"""
    try:
        output_dir = 'plots'
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        fig.write_html(output_path)
        logging.info(f"Plot saved to {output_path}")
    except Exception as e:
        logging.error(f"Error saving plot: {str(e)}")
        raise

def violin(column):
    """Main function to create and display violin plot for a specific column"""
    try:
        # Check if input file exists
        input_file = 'kidney_disease.csv'
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file {input_file} not found")
        
        # Load the data
        logging.info(f"Loading data from {input_file}")
        df = pd.read_csv(input_file)
        
        # Clean the data
        logging.info("Cleaning data...")
        df = clean_data(df)
        
        # Create and save violin plot
        fig = create_violin_plot(df, column)
        save_plot(fig, f'violin_{column}.html')
        
        # Display the plot
        return fig.show()
        
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        raise

# Example usage
if __name__ == "__main__":
    # Create violin plots for all numerical features
    numerical_features = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc',
        'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
    ]
    
    for feature in numerical_features:
        violin(feature) 