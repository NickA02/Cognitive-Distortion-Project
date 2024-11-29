import pandas as pd
import re

# Define valid classes
VALID_CLASSES = {
    'No Distortion',
    'Emotional Reasoning',
    'Overgeneralization',
    'Mental Filter',
    'Should Statements',
    'All-or-Nothing Thinking',
    'Mind Reading',
    'Fortune-telling',
    'Magnification',
    'Personalization',
    'Labeling'
}

def clean_class_name(class_name):
    """
    Cleans and standardizes class names.
    
    Parameters:
    class_name (str): The extracted class name
    
    Returns:
    str: Standardized class name or None if no match found
    """
    # Remove any leading/trailing whitespace
    cleaned = class_name.strip()
    
    # Try exact match first
    if cleaned in VALID_CLASSES:
        return cleaned
    
    # Try case-insensitive match
    for valid_class in VALID_CLASSES:
        if cleaned.lower() == valid_class.lower():
            return valid_class
            
    # Handle common variations
    if 'fortune telling' in cleaned.lower():
        return 'Fortune-telling'
    if 'all or nothing' in cleaned.lower():
        return 'All-or-Nothing Thinking'
        
    return None

def extract_final_classes(text):
    """
    Extracts and validates the final class predictions from a response text.
    
    Parameters:
    text (str): The response text containing class analysis and final predictions
    
    Returns:
    list: A list of validated class predictions
    """
    # Find all lines that start with "Class:" using regex
    classes = re.findall(r'Class: ([^\n]+)', text)
    
    # If no classes found using the simple pattern, try finding them in the final section
    if not classes:
        # Look for classes after specific phrases
        final_section = re.split(r'(?i)based on the above eliminations,|the most likely classes are:', text)
        if len(final_section) > 1:
            classes = re.findall(r'Class: ([^\n]+)', final_section[-1])
    
    # Clean and validate each class
    validated_classes = []
    for class_name in classes:
        cleaned_class = clean_class_name(class_name)
        if cleaned_class:
            validated_classes.append(cleaned_class)
    
    return validated_classes

def process_dataset(input_file, output_file):
    """
    Processes the dataset to extract and validate classes from the Response column.
    
    Parameters:
    input_file (str): Path to input CSV file
    output_file (str): Path to save the processed CSV file
    
    Returns:
    tuple: (processed DataFrame, analysis dictionary)
    """
    # Read the dataset
    df = pd.read_csv(input_file)
    
    # Extract classes for each response
    df['extracted_classes'] = df['Response'].apply(extract_final_classes)
    
    # Create separate columns for each class
    df['Class_1'] = df['extracted_classes'].apply(lambda x: x[0] if len(x) > 0 else None)
    df['Class_2'] = df['extracted_classes'].apply(lambda x: x[1] if len(x) > 1 else None)
    
    # Create analysis summary
    analysis = {
        'total_rows': len(df),
        'rows_with_classes': len(df[df['Class_1'].notna()]),
        'rows_with_two_classes': len(df[df['Class_2'].notna()]),
        'class_distribution': df['Class_1'].value_counts().to_dict(),
        'class_pairs': df[df['Class_2'].notna()].groupby(['Class_1', 'Class_2']).size().to_dict()
    }
    
    # Drop the intermediate extracted_classes column
    df = df.drop('extracted_classes', axis=1)
    
    # Save the processed dataset
    df.to_csv(output_file, index=False)
    
    return df, analysis

def print_analysis(analysis):
    """
    Prints a formatted analysis of the processing results.
    """
    print(f"\nDataset Analysis:")
    print(f"Total rows processed: {analysis['total_rows']}")
    print(f"Rows with at least one class: {analysis['rows_with_classes']}")
    print(f"Rows with two classes: {analysis['rows_with_two_classes']}")
    
    print("\nClass Distribution:")
    for class_name, count in sorted(analysis['class_distribution'].items()):
        print(f"  {class_name}: {count}")
    
    print("\nMost Common Class Pairs:")
    sorted_pairs = sorted(analysis['class_pairs'].items(), key=lambda x: x[1], reverse=True)
    for (class1, class2), count in sorted_pairs[:5]:
        print(f"  {class1} + {class2}: {count}")

# Example usage:
processed_df, analysis = process_dataset('/Users/ulugsali/Desktop/Cognitive-Distortion-Project/results/multiclass/elimination/llama3.2-3b/zero-shot.csv', 'processed_dataset.csv')

# Print analysis
print_analysis(analysis)

# Check for any rows where classification might have failed
print("\nRows with missing classifications:")
print(processed_df[processed_df['Class_1'].isna()][['id', 'Response']])