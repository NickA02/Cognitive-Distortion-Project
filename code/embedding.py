import ollama
import chromadb
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

class LlamaDistortionClassifier:
    def __init__(self):
        self.base_prompt = """A cognitive distortion is defined as an unreasonable biased thought that create negative patterns in the way a person thinks. Your job is to label therapy patient anecdotes with the following labels:
Labels:
    - No Distortion -- Having a realistic, reasonably biased recollection in the anecdote. Any negative thought seems to be based mostly on plausible scenarios.
    - Emotional Reasoning -- Believing "I feel that way, so it must be true"
    - Overgeneralization -- Drawing sweeping negative conclusions with limited instances
    - Mental Filter -- Focusing only on limited negative aspects and not the excessive positive ones.
    - Should Statements -- Expecting things or personal behavior should be a certain way.
    - All-or-Nothing Thinking -- Binary thought pattern. Considering anything short of perfection as a failure.
    - Mind Reading -- Concluding that others are reacting negatively to you, without any basis in fact.
    - Fortune-telling -- Predicting that an event will always result in the worst possible outcome.
    - Magnification -- Exaggerating or Catastrophizing the outcome of certain events or behavior.
    - Personalization -- Holding oneself personally responsible for events beyond one's control.
    - Labeling -- Attaching labels to oneself or others (ex: "loser", "perfect").

Here are some relevant examples:

{examples}

The following input is from a therapy session snippet.
Identify the Distortion present.
If there is no distortion, just respond 'No Distortion'. There should be no explanation, only the label.

Patient Question: """
        
        self.collection = None
        self.setup_database()

    def setup_database(self):
        """Initialize ChromaDB"""
        client = chromadb.Client()
        try:
            client.delete_collection("cognitive_distortions")
        except:
            pass
        self.collection = client.create_collection(name="cognitive_distortions")

    def train(self, train_df):
        """Add training examples to ChromaDB"""
        print("Adding training examples to database...")
        for idx, row in train_df.iterrows():
            if pd.isna(row['Patient Question']) or pd.isna(row['Dominant Distortion']):
                continue

            #getting embeddings for the text
            response = ollama.embeddings(
                model="mxbai-embed-large",
                prompt=row['Patient Question']
            )

            #storing in ChromaDB
            self.collection.add(
                ids=[str(idx)],
                embeddings=[response["embedding"]],
                documents=[row['Patient Question']],
                metadatas=[{"distortion": row['Dominant Distortion']}]
            )

    def get_similar_examples(self, text, n_examples=3):
        """Find similar examples using embeddings"""
        response = ollama.embeddings(
            model="mxbai-embed-large",
            prompt=text
        )

        results = self.collection.query(
            query_embeddings=[response["embedding"]],
            n_results=n_examples
        )

        examples_text = ""
        for doc, metadata in zip(results['documents'][0], results['metadatas'][0]):
            examples_text += f"Example text: {doc}\n"
            examples_text += f"Distortion: {metadata['distortion']}\n\n"

        return examples_text

    def predict(self, text, use_rag=True):
        """Predict distortion with option to use RAG"""
        if use_rag:
            # Get similar examples
            examples = self.get_similar_examples(text)
            # Create prompt with examples
            prompt = self.base_prompt.format(examples=examples) + text
        else:
            # Zero-shot without examples
            prompt = self.base_prompt.format(examples="") + text

        # Generate prediction
        response = ollama.generate(
            model="llama2:3.2-7b",
            prompt=prompt,
            temperature=0.1
        )

        return response['response'].strip()

    def evaluate(self, test_df, use_rag=True):
        """Evaluate model on test set with detailed metrics"""
        print(f"\nEvaluating model using {'RAG' if use_rag else 'Zero-shot'} approach...")
        
        predictions = []
        actuals = []
        
        for idx, row in test_df.iterrows():
            if pd.isna(row['Patient Question']) or pd.isna(row['Dominant Distortion']):
                continue
                
            prediction = self.predict(row['Patient Question'], use_rag=use_rag)
            predictions.append(prediction)
            actuals.append(row['Dominant Distortion'])
            
            # Print progress every 10 examples
            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1} test examples...")
        
        # Get classification report as dict for easier metric extraction
        report = classification_report(actuals, predictions, output_dict=True)
        
        # Print per-class F1 scores
        print("\nPer-class F1 scores:")
        print("-" * 50)
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                print(f"{class_name}: {metrics['f1-score']:.3f}")
        
        # Print overall metrics
        print("\nOverall Metrics:")
        print("-" * 50)
        print(f"Accuracy: {report['accuracy']:.3f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.3f}")
        print(f"Weighted F1: {report['weighted avg']['f1-score']:.3f}")
        
        # Print confusion matrix
        print("\nConfusion Matrix:")
        print(confusion_matrix(actuals, predictions))
        
        # Save results to CSV
        results_df = pd.DataFrame({
            'Text': test_df['Patient Question'],
            'Actual': actuals,
            'Predicted': predictions
        })
        
        # Save detailed metrics
        metrics_dict = {
            'Class': [],
            'F1': [],
            'Precision': [],
            'Recall': []
        }
        
        for class_name, metrics in report.items():
            if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
                metrics_dict['Class'].append(class_name)
                metrics_dict['F1'].append(metrics['f1-score'])
                metrics_dict['Precision'].append(metrics['precision'])
                metrics_dict['Recall'].append(metrics['recall'])
        
        metrics_df = pd.DataFrame(metrics_dict)
        
        # Save both results and metrics
        results_df.to_csv('prediction_results.csv', index=False)
        metrics_df.to_csv('metrics_by_class.csv', index=False)
        
        print("\nResults and metrics saved to prediction_results.csv and metrics_by_class.csv")
        
        return predictions, actuals, report

def main():
    print("Loading datasets...")
    train_df = pd.read_csv('datasets/train.csv')
    test_df = pd.read_csv('datasets/test.csv')
    
    # Initialize classifier
    classifier = LlamaDistortionClassifier()
    
    # Train the model (add examples to database)
    classifier.train(train_df)
    
    # Evaluate both approaches on the test set
    
    # 1. Zero-shot evaluation
    print("\nEvaluating Zero-shot approach...")
    zero_shot_preds, actuals, zero_shot_metrics = classifier.evaluate(test_df, use_rag=False)
    
    # 2. RAG evaluation
    print("\nEvaluating RAG approach...")
    rag_preds, actuals, rag_metrics = classifier.evaluate(test_df, use_rag=True)
    
    # Compare approaches
    print("\nComparison of Approaches:")
    print("-" * 50)
    print(f"Zero-shot Accuracy: {zero_shot_metrics['accuracy']:.3f}")
    print(f"RAG Accuracy: {rag_metrics['accuracy']:.3f}")
    print(f"Zero-shot Macro F1: {zero_shot_metrics['macro avg']['f1-score']:.3f}")
    print(f"RAG Macro F1: {rag_metrics['macro avg']['f1-score']:.3f}")

if __name__ == "__main__":
    main()