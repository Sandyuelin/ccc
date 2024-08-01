from PIL import Image, ImageDraw, ImageFont

def render_character(character, font_path, output_image_path, image_size=(64, 64)):
    # Create a blank white image
    image = Image.new('RGB', image_size, 'white')
    draw = ImageDraw.Draw(image)

    # Load the font
    font = ImageFont.truetype(font_path, size=48)

    # Calculate text size and position
    text_size = draw.textsize(character, font=font)
    text_position = ((image_size[0] - text_size[0]) / 2, (image_size[1] - text_size[1]) / 2)

    # Draw the character
    draw.text(text_position, character, fill='black', font=font)

    # Save the image
    image.save(output_image_path)
