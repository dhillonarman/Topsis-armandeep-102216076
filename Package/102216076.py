import sys
import pandas as pd
import numpy as np

def validate_inputs(weights, impacts, num_columns):
    """Validate weights and impacts."""
    # Convert weights to a list of numbers
    try:
        weights = [float(w) for w in weights.split(",")]
    except ValueError:
        raise ValueError("Weights must be numeric and comma-separated.")

    # Convert impacts to a list and validate
    impacts = impacts.split(",")
    if not all(i in ["+", "-"] for i in impacts):
        raise ValueError("Impacts must be either '+' or '-' and comma-separated.")

    # Check if the number of weights, impacts, and numeric columns match
    if len(weights) != num_columns or len(impacts) != num_columns:
        raise ValueError("Number of weights, impacts, and numeric columns must be the same.")

    return np.array(weights), np.array(impacts)

def topsis(input_file, weights, impacts, output_file):
    """Perform TOPSIS on the input data."""
    try:
        # Read CSV file
        df = pd.read_csv(input_file)

        # Ensure at least 3 columns
        if df.shape[1] < 3:
            raise ValueError("Input file must contain at least three columns.")

        # Extract numeric values (excluding the first column)
        data = df.iloc[:, 1:].values

        # Ensure all numeric values
        if not np.issubdtype(data.dtype, np.number):
            raise ValueError("All columns (except the first) must contain numeric values.")

        # Validate weights and impacts
        weights, impacts = validate_inputs(weights, impacts, data.shape[1])

        # **STEP 1: Normalize the matrix**
        norm_data = data / np.sqrt((data**2).sum(axis=0))

        # **STEP 2: Multiply by weights**
        weighted_data = norm_data * weights

        # **STEP 3: Determine ideal best and worst solutions**
        ideal_best = np.where(impacts == "+", weighted_data.max(axis=0), weighted_data.min(axis=0))
        ideal_worst = np.where(impacts == "+", weighted_data.min(axis=0), weighted_data.max(axis=0))

        # **STEP 4: Compute Euclidean distances**
        dist_best = np.sqrt(((weighted_data - ideal_best)**2).sum(axis=1))
        dist_worst = np.sqrt(((weighted_data - ideal_worst)**2).sum(axis=1))

        # **STEP 5: Compute TOPSIS scores**
        scores = dist_worst / (dist_best + dist_worst)

        # **STEP 6: Rank alternatives**
        df["Topsis Score"] = scores
        df["Rank"] = df["Topsis Score"].rank(ascending=False, method="dense").astype(int)

        # Save to output CSV
        df.to_csv(output_file, index=False)
        print(f"TOPSIS computation successful! Results saved to {output_file}")

    except FileNotFoundError:
        print("Error: Input file not found.")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python <RollNumber>.py <InputDataFile> <Weights> <Impacts> <ResultFileName>")
        sys.exit(1)

    input_file = sys.argv[1]
    weights = sys.argv[2]
    impacts = sys.argv[3]
    output_file = sys.argv[4]

    topsis(input_file, weights, impacts, output_file)
