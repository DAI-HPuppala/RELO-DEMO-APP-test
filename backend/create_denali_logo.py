#!/usr/bin/env python3
"""Create a Denali logo test image for VLM warmup"""

from PIL import Image, ImageDraw, ImageFont
import os

def create_denali_logo():
    """Create a professional Denali company logo for VLM warmup testing"""
    
    # Create a high-quality image (1024x768 for VLM optimization)
    img = Image.new('RGB', (1024, 768), color='#f8f9fa')  # Light background
    draw = ImageDraw.Draw(img)
    
    # Draw Denali company branding
    # Background gradient effect
    for y in range(768):
        gradient_color = int(248 - (y / 768) * 20)  # Subtle gradient
        draw.line([(0, y), (1024, y)], fill=(gradient_color, gradient_color + 2, gradient_color + 5))
    
    # Main logo area
    logo_rect = [200, 150, 824, 450]
    draw.rectangle(logo_rect, fill='#1e3a5f', outline='#2c5282', width=4)
    
    # Company name "DENALI" - simulate text
    try:
        # Try to use a better font if available
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 72)
        font_medium = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 28)
    except:
        # Fallback to default font
        font_large = ImageFont.load_default()
        font_medium = ImageFont.load_default()
        font_small = ImageFont.load_default()
    
    # Main company name
    draw.text((350, 220), "DENALI", font=font_large, fill='white', anchor="mm")
    
    # Subtitle
    draw.text((350, 300), "Advanced Integration Systems", font=font_medium, fill='#e2e8f0', anchor="mm")
    
    # Additional company details for VLM analysis
    draw.text((350, 350), "Returns Classification Technology", font=font_small, fill='#cbd5e0', anchor="mm")
    
    # Add some geometric elements for visual interest
    # Left accent
    draw.polygon([(220, 170), (280, 200), (220, 230)], fill='#4299e1')
    # Right accent  
    draw.polygon([(804, 170), (744, 200), (804, 230)], fill='#4299e1')
    
    # Bottom section with additional text for comprehensive VLM testing
    bottom_rect = [100, 500, 924, 650]
    draw.rectangle(bottom_rect, fill='white', outline='#e2e8f0', width=2)
    
    # Add sample clothing item for VLM context
    draw.text((512, 530), "AI-Powered Garment Classification", font=font_medium, fill='#2d3748', anchor="mm")
    draw.text((512, 570), "• Real-time Analysis  • Damage Detection  • Brand Recognition", font=font_small, fill='#4a5568', anchor="mm")
    draw.text((512, 610), "TEST IMAGE - VLM Warmup & GPU Optimization", font=font_small, fill='#718096', anchor="mm")
    
    # Add some sample garment elements for classification testing
    # Simulate a simple shirt outline
    shirt_x, shirt_y = 150, 520
    shirt_points = [
        (shirt_x, shirt_y + 20), (shirt_x + 15, shirt_y), (shirt_x + 45, shirt_y),
        (shirt_x + 60, shirt_y + 20), (shirt_x + 60, shirt_y + 80),
        (shirt_x, shirt_y + 80)
    ]
    draw.polygon(shirt_points, fill='#3182ce', outline='#2c5aa0', width=2)
    draw.text((shirt_x + 30, shirt_y + 90), "Sample Item", font=font_small, fill='#4a5568', anchor="mm")
    
    return img

if __name__ == "__main__":
    # Create the logo
    logo = create_denali_logo()
    
    # Save to assets directory
    assets_dir = os.path.join(os.path.dirname(__file__), 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    logo_path = os.path.join(assets_dir, 'denali_logo.png')
    logo.save(logo_path, 'PNG', quality=95)
    
    print(f"✅ Denali logo created successfully: {logo_path}")
    print(f"📏 Image size: {logo.size}")
    print("🎯 Ready for VLM warmup testing")