from transformers import LlamaTokenizer, LlamaForCausalLM
import torch

# Function to load the LLaMA model and tokenizer
def load_model(model_name="meta-llama/Llama-2-13b"):
    print("Loading model and tokenizer...")
    tokenizer = LlamaTokenizer.from_pretrained(model_name)
    model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    print("Model and tokenizer loaded successfully!")
    return tokenizer, model

# Function to generate a social media marketing workflow
def generate_workflow(model, tokenizer, business_details, max_tokens=300):
    # Define the prompt with client-specific details
    prompt = f"""
    Generate a detailed social media marketing and branding workflow for the following business:
    {business_details}
    Include:
    1. Content themes
    2. Posting schedules
    3. Platform strategies
    """
    
    # Tokenize input and generate output
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# Main function to handle user input and generate the workflow
def main():
    # Specify the LLaMA model to use
    model_name = "meta-llama/Llama-2-13b"  # Update this if using another LLaMA variant

    # Load the model and tokenizer
    tokenizer, model = load_model(model_name)

    # Input details about the client's business
    print("Enter details about the client's business.")
    business_type = input("Business Type (e.g., Vegan Restaurant): ")
    target_audience = input("Target Audience (e.g., Health-conscious millennials): ")
    goals = input("Goals (e.g., Increase Instagram engagement): ")

    # Combine the input details
    business_details = f"""
    - Business Type: {business_type}
    - Target Audience: {target_audience}
    - Goals: {goals}
    """

    # Generate the marketing workflow
    print("\nGenerating workflow...")
    workflow = generate_workflow(model, tokenizer, business_details)
    print("\nGenerated Marketing Workflow:")
    print(workflow)

# Run the program
if __name__ == "__main__":
    main()
