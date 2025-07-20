#!/usr/bin/env python3
"""
Demo script showing the new multimodal response improvements
"""
import json
import datetime
import os

# Simulate the new feedback logging functionality
class JarvisDemo:
    def __init__(self):
        self.generation_params = {
            "max_new_tokens": 150,
            "num_beams": 3,
            "temperature": 0.7,
            "do_sample": True
        }
        self.feedback_data = []
    
    def log_interaction(self, image_path, prompt, response, feedback=None):
        """Log interaction for fine-tuning data collection"""
        interaction = {
            "timestamp": datetime.datetime.now().isoformat(),
            "image_path": image_path,
            "prompt": prompt,
            "response": response,
            "feedback": feedback,
            "generation_params": self.generation_params.copy()
        }
        self.feedback_data.append(interaction)
        print(f"✓ Logged interaction with feedback: {feedback}")
    
    def update_generation_params(self, **kwargs):
        """Update generation parameters"""
        self.generation_params.update(kwargs)
        print(f"✓ Updated parameters: {kwargs}")
    
    def save_feedback_data(self):
        """Save feedback data to JSON file"""
        if self.feedback_data:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"demo_feedback_data_{timestamp}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(self.feedback_data, f, indent=2, ensure_ascii=False)
            print(f"✓ Saved {len(self.feedback_data)} interactions to {filename}")
            return filename
        return None
    
    def simulate_response_generation(self, image_path, prompt):
        """Simulate response generation with current parameters"""
        print(f"🤖 Generating response with parameters:")
        print(f"   - max_new_tokens: {self.generation_params['max_new_tokens']}")
        print(f"   - num_beams: {self.generation_params['num_beams']}")
        print(f"   - temperature: {self.generation_params['temperature']}")
        print(f"   - do_sample: {self.generation_params['do_sample']}")
        
        # Simulate different response quality based on parameters
        if self.generation_params['num_beams'] >= 5:
            return "High quality response with excellent beam search"
        elif self.generation_params['max_new_tokens'] >= 200:
            return "Detailed response with extended token length"
        else:
            return "Standard response with default parameters"

def main():
    print("🚀 Jarvis Multimodal Response Improvements Demo\n")
    
    demo = JarvisDemo()
    
    # Demo 1: Parameter experimentation
    print("📊 Demo 1: Parameter Experimentation")
    print("=" * 50)
    
    scenarios = [
        {"max_new_tokens": 100, "num_beams": 1, "description": "Fast, low quality"},
        {"max_new_tokens": 200, "num_beams": 5, "description": "High quality, slower"},
        {"max_new_tokens": 150, "num_beams": 3, "temperature": 0.9, "description": "Balanced with creativity"}
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\nScenario {i}: {scenario.pop('description')}")
        demo.update_generation_params(**scenario)
        response = demo.simulate_response_generation("test_image.png", "Describe this screen")
        print(f"Response: {response}")
        demo.log_interaction("test_image.png", "Describe this screen", response)
    
    # Demo 2: Feedback collection
    print(f"\n📝 Demo 2: Feedback Collection")
    print("=" * 50)
    
    # Simulate user feedback
    feedbacks = ["good", "bad", "good"]
    for i, feedback in enumerate(feedbacks):
        if i < len(demo.feedback_data):
            demo.feedback_data[i]["feedback"] = feedback
            print(f"✓ Added {feedback} feedback to interaction {i+1}")
    
    # Demo 3: Data export for fine-tuning
    print(f"\n💾 Demo 3: Training Data Export")
    print("=" * 50)
    
    filename = demo.save_feedback_data()
    if filename:
        # Show sample of exported data
        with open(filename, 'r') as f:
            data = json.load(f)
        
        print(f"Sample training data structure:")
        if data:
            sample = data[0]
            print(json.dumps({k: v for k, v in sample.items() if k != 'prompt'}, indent=2))
    
    print(f"\n✅ Demo Complete!")
    print("New features demonstrated:")
    print("  ✓ Configurable generation parameters")
    print("  ✓ Real-time parameter updates")
    print("  ✓ Feedback collection system")
    print("  ✓ Training data export")
    print("  ✓ Parameter persistence")
    
    # Clean up demo file
    if filename and os.path.exists(filename):
        os.remove(filename)
        print(f"  ✓ Cleaned up demo file: {filename}")

if __name__ == "__main__":
    main()