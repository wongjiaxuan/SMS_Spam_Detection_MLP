"""
Generate Training Loss Curve Figure
"""

import matplotlib.pyplot as plt
import joblib
import numpy as np
import os

# Create figures directory if it doesn't exist
os.makedirs('reports/figures', exist_ok=True)

print("Loading model...")
try:
    model = joblib.load('models/spam_detector_model.pkl')
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# Get training loss history
if hasattr(model, 'loss_curve_'):
    loss_values = model.loss_curve_
    print(f"✅ Found {len(loss_values)} loss values")
else:
    print("❌ No loss curve found in model")
    exit(1)

# Create the figure
plt.figure(figsize=(10, 6))

# Plot loss curve
plt.plot(loss_values, linewidth=2, color='#3498db', label='Training Loss')

# Add labels and title
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('Training Loss Curve', fontsize=16, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)

# Add text annotation for final values
final_loss = loss_values[-1]
plt.annotate(f'Final Loss: {final_loss:.4f}',
             xy=(len(loss_values)-1, final_loss),
             xytext=(len(loss_values)-5, final_loss + 0.02),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

# Show final iteration
plt.annotate(f'Iteration: {len(loss_values)}',
             xy=(len(loss_values)-1, final_loss),
             xytext=(len(loss_values)-3, final_loss - 0.05),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

plt.tight_layout()

# Save figure
output_path = 'reports/figures/training_loss_curve.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"✅ Figure saved to: {output_path}")

# Also show the plot
plt.show()

print("\n" + "="*50)
print("TRAINING LOSS CURVE SUMMARY")
print("="*50)
print(f"  Iterations: {len(loss_values)}")
print(f"  Final Loss: {final_loss:.4f}")
print(f"  Figure saved: {output_path}")